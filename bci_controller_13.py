
import collections
import os
import keyboard
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, welch, iirnotch, filtfilt
import torch
import torch.nn as nn
import threading
from collections import deque
import pickle
import socket
import json
import logging
import datetime
import pyautogui
import subprocess
import io
import sys
import importlib
from pynput.keyboard import Key, Controller

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bci_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BCI_Controller")

# 配置參數
MODEL_PATH = r"C:\Users\albio\success4.pt"  # 默認路徑，需要時將更新
SAMPLING_RATE = 500  # 500 Hz
WINDOW_SIZE = 20.0  # 10秒窗口進行分析
WINDOW_OVERLAP = 0  # 窗口之間75%重疊
BUFFER_SIZE = int(WINDOW_SIZE * SAMPLING_RATE)  # 緩衝區大小（樣本數）
STEP_SIZE = int(BUFFER_SIZE * (1 - WINDOW_OVERLAP))  # 步長（樣本數）

# 初始化鍵盤控制器
keyboard_controller = Controller()

# 定義頻率帶用於特徵提取
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'high_beta': (20, 30),
    'gamma': (30, 50)
}

# 控制映射
STATE_TO_ACTION = {
    0:"down",
    1:"up",
    2:"left",
    3:"right",
}

# 類別映射用於顯示
CLASS_MAPPING = {
    0: "放鬆 (Alpha)",
    1: "專注 (Beta)",
    2: "壓力 (High Beta)",
    3: "記憶 (Theta)"
}

# 從use_ensemble_model.txt直接加載類別定義
class ScalerWrapper:
    """用於使sklearn縮放器可與PyTorch模型一起序列化的包裝器"""
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

class SelectorWrapper:
    """用於使sklearn選擇器可與PyTorch模型一起序列化的包裝器"""
    def __init__(self, selector):
        self.selector = selector

    def fit(self, X, y=None):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

# 自定義Unpickler用於安全加載
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 處理我們知道安全的自定義類
        if module == "__main__" or module == "builtins":
            if name == "ScalerWrapper":
                return ScalerWrapper
            elif name == "SelectorWrapper":
                return SelectorWrapper

        # 對於其他類，使用默認行為
        return super().find_class(module, name)

# 從use_ensemble_model.txt直接加載模型類定義
class EEGFeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(EEGFeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()

        # Add output_dim attribute for compatibility
        self.output_dim = num_classes

    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu(self.fc2(x))))
        x = self.dropout3(self.bn3(self.relu(self.fc3(x))))
        x = self.fc4(x)
        return nn.functional.softmax(x, dim=1)

class MetaLearner(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(MetaLearner, self).__init__()
        # 擴展的架構與更好的正則化
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate * 0.8)

        self.output = nn.Linear(32, num_classes)

        # 用於校準的溫度參數（初始為1.0）
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)

        x = self.leaky_relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)

        x = self.leaky_relu(self.bn3(self.layer3(x)))
        x = self.dropout3(x)

        x = self.output(x)

        # 應用溫度縮放以獲得更好的校準
        return x / self.temperature

class PyTorchEnsembleModel(nn.Module):
    def __init__(self, base_models, selectors, scalers, meta_learner, num_classes):
        """
        Initialize PyTorch Ensemble Model with improved prediction handling

        Parameters:
        -----------
        base_models : list
            List of PyTorch base models
        selectors : list
            List of feature selectors for each base model
        scalers : list
            List of feature scalers for each base model
        meta_learner : torch.nn.Module
            Meta-learner model that combines predictions from base models
        num_classes : int
            Number of output classes
        """
        super(PyTorchEnsembleModel, self).__init__()

        self.base_models = nn.ModuleList(base_models)
        self.selectors = selectors
        self.scalers = scalers
        self.meta_learner = meta_learner
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass through the ensemble model with improved error handling

        Parameters:
        -----------
        x : torch.Tensor or numpy.ndarray
            Input features

        Returns:
        --------
        torch.Tensor
            Final predictions from meta-learner
        """
        # Get predictions from all base models
        base_predictions = []

        for model_idx, (model, selector, scaler) in enumerate(zip(self.base_models, self.selectors, self.scalers)):
            try:
                x_model = x.clone().detach().numpy() if isinstance(x, torch.Tensor) else x.copy()

                # Apply feature selection FIRST, then scaling
                if selector is not None:
                    x_model = selector.transform(x_model)

                # Apply scaling AFTER selection
                if scaler is not None:
                    x_model = scaler.transform(x_model)

                # Convert to tensor if needed
                if not isinstance(x_model, torch.Tensor):
                    x_model = torch.FloatTensor(x_model)

                # Get predictions from base model
                model.eval()
                with torch.no_grad():
                    base_pred = model(x_model)

                base_predictions.append(base_pred)
            except Exception as e:
                logger.error(f"Error in base model {model_idx}: {e}")
                # Create empty predictions of appropriate shape if model fails
                if len(base_predictions) > 0:
                    # Use the shape from previous successful model
                    empty_pred = torch.zeros_like(base_predictions[0])
                else:
                    # Estimate shape based on number of classes
                    empty_pred = torch.zeros((x.shape[0] if hasattr(x, 'shape') else 1, self.num_classes))
                base_predictions.append(empty_pred)

        # Concatenate predictions from all base models
        meta_features = torch.cat(base_predictions, dim=1)

        # Make final prediction using meta-learner
        self.meta_learner.eval()
        final_pred = self.meta_learner(meta_features)

        return final_pred

    def predict(self, X):
        """
        Make predictions with the ensemble model

        Parameters:
        -----------
        X : numpy.ndarray
            Input features

        Returns:
        --------
        numpy.ndarray
            Predicted probabilities
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.forward(X_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.numpy()

# 創建AHK腳本文件的函數
def create_ahk_scripts():
    """創建必要的AHK腳本文件"""
    try:
        # 獲取當前目錄
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 創建上鍵腳本
        up_script_path = os.path.join(current_dir, "key_up.ahk")
        with open(up_script_path, "w") as f:
            f.write("""#SingleInstance Force
Sleep 200
Send "{Up}"
ExitApp
""")
        logger.info(f"創建了AHK腳本: {up_script_path}")

        # 創建下鍵腳本
        down_script_path = os.path.join(current_dir, "key_down.ahk")
        with open(down_script_path, "w") as f:
            f.write("""#SingleInstance Force
Sleep 200
Send "{Down}"
ExitApp
""")
        logger.info(f"創建了AHK腳本: {down_script_path}")

        # 創建左鍵腳本
        left_script_path = os.path.join(current_dir, "key_left.ahk")
        with open(left_script_path, "w") as f:
            f.write("""#SingleInstance Force
Sleep 200
Send "{Left}"
ExitApp
""")
        logger.info(f"創建了AHK腳本: {left_script_path}")

        # 創建右鍵腳本
        right_script_path = os.path.join(current_dir, "key_right.ahk")
        with open(right_script_path, "w") as f:
            f.write("""#SingleInstance Force
Sleep 200
Send "{Right}"
ExitApp
""")
        logger.info(f"創建了AHK腳本: {right_script_path}")

        return True
    except Exception as e:
        logger.error(f"創建AHK腳本時出錯: {e}")
        return False

# 運行AHK腳本的函數
def run_ahk_script(key_name):
    """執行AHK腳本按下指定的按鍵。"""
    if not AHK_AVAILABLE:
        logger.warning("AutoHotkey不可用。無法執行AHK腳本。")
        return False

    try:
        # 獲取當前目錄
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 根據方向鍵選擇相應的腳本
        script_name = f"key_{key_name}.ahk"
        ahk_script = os.path.join(current_dir, script_name)

        if not os.path.exists(ahk_script):
            logger.error(f"AHK腳本未找到: {ahk_script}")
            # 嘗試創建腳本
            if not create_ahk_scripts():
                return False
            if not os.path.exists(ahk_script):
                return False

        # 執行腳本並記錄完整命令
        cmd = [AHK_PATH, ahk_script]
        logger.info(f"執行命令: {' '.join(cmd)}")

        # 使用subprocess捕獲完整輸出
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"AHK執行結果: {result.returncode}")
        if result.stdout:
            logger.info(f"AHK輸出: {result.stdout}")
        if result.stderr:
            logger.warning(f"AHK錯誤: {result.stderr}")

        logger.info(f"成功執行AHK腳本按下按鍵: {key_name}")
        return True
    except Exception as e:
        logger.error(f"執行AHK腳本時出錯: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# 測試AHK功能的函數
def test_ahk():
    """測試AHK腳本是否可以正常工作"""
    logger.info("測試AHK腳本...")

    # 創建測試腳本
    create_ahk_scripts()

    # 測試腳本
    result = run_ahk_script("up")

    if result:
        logger.info("AHK測試成功！")
    else:
        logger.error("AHK測試失敗。")

    return result

# 手動觸發狀態的函數（用於測試）
def manually_trigger_state(processor, state):
    """手動觸發特定狀態（用於測試）"""
    if state in STATE_TO_ACTION:
        logger.info(f"手動觸發狀態: {CLASS_MAPPING.get(state, f'狀態 {state}')}")
        processor.trigger_action(state)
        return True
    else:
        logger.warning(f"狀態 {state} 沒有映射到動作")
        return False

# 直接從use_ensemble_model.txt引入特徵提取函數
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def extract_band_powers(data, fs):
    """
    Extract power in different frequency bands
    """
    # Calculate power spectral density
    freqs, psd = welch(data, fs, nperseg=min(1024, len(data)))

    # Calculate power in each frequency band
    powers = {}
    for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
        # Find indices corresponding to the frequency band
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        # Calculate power in the band
        powers[band_name] = np.mean(psd[idx_band])

    return powers

def extract_features(data, fs):
    """
    Extract comprehensive features from EEG segment with more advanced measures
    """
    features = {}

    # Apply bandpass filter for cleaner signal (1-45 Hz)
    filtered_data = bandpass_filter(data, 1, 45, fs)

    # Time-domain features
    features['mean'] = np.mean(filtered_data)
    features['std'] = np.std(filtered_data)
    features['kurtosis'] = np.mean((filtered_data - np.mean(filtered_data))**4) / (np.std(filtered_data)**4) if np.std(filtered_data) > 0 else 0
    features['skewness'] = np.mean((filtered_data - np.mean(filtered_data))**3) / (np.std(filtered_data)**3) if np.std(filtered_data) > 0 else 0
    features['activity'] = np.var(filtered_data)
    features['mobility'] = np.sqrt(np.var(np.diff(filtered_data)) / features['activity']) if features['activity'] > 0 else 0

    # Complexity (measure of signal complexity)
    if features['mobility'] > 0:
        var_d2 = np.var(np.diff(np.diff(filtered_data)))
        var_d1 = np.var(np.diff(filtered_data))
        features['complexity'] = np.sqrt(var_d2 / var_d1) / features['mobility'] if var_d1 > 0 else 0
    else:
        features['complexity'] = 0

    # Peak-to-peak amplitude
    features['p2p_amp'] = np.max(filtered_data) - np.min(filtered_data)

    # Root mean square
    features['rms'] = np.sqrt(np.mean(filtered_data**2))

    # Absolute mean values for each segment
    segment_length = len(filtered_data) // 4
    for i in range(4):
        segment = filtered_data[i*segment_length:(i+1)*segment_length]
        features[f'abs_mean_seg{i+1}'] = np.mean(np.abs(segment))

    # Zero-crossing rate
    zero_crossings = np.where(np.diff(np.signbit(filtered_data)))[0]
    features['zero_cross_rate'] = len(zero_crossings) / len(filtered_data)

    # Frequency-domain features from predefined bands
    band_powers = extract_band_powers(filtered_data, fs)
    for band, power in band_powers.items():
        features[f'{band}_power'] = power

    # Normalized band powers
    total_power = sum(power for power in band_powers.values())
    if total_power > 0:
        for band, power in band_powers.items():
            features[f'{band}_power_norm'] = power / total_power

    # Power ratios (important for EEG analysis)
    if band_powers['beta'] > 0:
        features['alpha_beta_ratio'] = band_powers['alpha'] / band_powers['beta']
        features['theta_beta_ratio'] = band_powers['theta'] / band_powers['beta']
        features['delta_beta_ratio'] = band_powers['delta'] / band_powers['beta']
    else:
        features['alpha_beta_ratio'] = 0
        features['theta_beta_ratio'] = 0
        features['delta_beta_ratio'] = 0

    if band_powers['alpha'] > 0:
        features['theta_alpha_ratio'] = band_powers['theta'] / band_powers['alpha']
        features['delta_alpha_ratio'] = band_powers['delta'] / band_powers['alpha']
    else:
        features['theta_alpha_ratio'] = 0
        features['delta_alpha_ratio'] = 0

    # Additional statistical features
    # Median absolute deviation
    features['mad'] = np.median(np.abs(filtered_data - np.median(filtered_data)))

    # Statistical moments
    features['1st_moment'] = np.mean(filtered_data)  # Mean (already calculated)
    features['2nd_moment'] = np.mean(filtered_data**2)
    features['3rd_moment'] = np.mean(filtered_data**3)
    features['4th_moment'] = np.mean(filtered_data**4)

    return features

def preprocess_eeg_data(file_path):
    """
    Preprocess EEG data from a single txt file with advanced processing
    """
    # Read data
    try:
        data = np.loadtxt(file_path)
        logger.info(f"Successfully loaded {file_path}, shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None, None

    # Check for NaN or infinity values
    if np.isnan(data).any() or np.isinf(data).any():
        logger.warning(f"Warning: {file_path} contains NaN or infinity values. Cleaning...")
        data = np.nan_to_num(data)

    # Calculate segment length in samples (use all data)
    segment_length = len(data)

    # Extract segment and process
    segments = [data]
    features_list = []

    # Apply a notch filter to remove 50/60 Hz power line interference
    notch_freq = 50  # For 50 Hz power line noise (or use 60 Hz depending on location)
    quality_factor = 30  # Quality factor
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, SAMPLING_RATE)
    segment_filtered = filtfilt(b_notch, a_notch, data)

    # Apply bandpass filter (1-45 Hz to focus on relevant EEG frequencies)
    segment_filtered = bandpass_filter(segment_filtered, 1, 45, SAMPLING_RATE)

    # Extract features
    features = extract_features(segment_filtered, SAMPLING_RATE)
    features_list.append(features)

    return np.array(segments), features_list

def predict_from_txt(file_path, model):
    """
    Make predictions from an EEG text file using the ensemble model

    Parameters:
    -----------
    file_path : str
        Path to the EEG text file
    model : torch.nn.Module
        The loaded model

    Returns:
    --------
    tuple
        (predicted_classes, predicted_probabilities)
    """
    # Preprocess EEG data
    segments, features_list = preprocess_eeg_data(file_path)
    if not features_list:
        raise ValueError("No features extracted from the file.")

    # Convert to feature matrix
    feature_names = list(features_list[0].keys())
    X_features = np.array([[f[name] for name in feature_names] for f in features_list])

    # Make predictions
    try:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_features)
            outputs = model(X_tensor)
            predicted_probs = torch.nn.functional.softmax(outputs, dim=1).numpy()
            predicted_classes = np.argmax(predicted_probs, axis=1)
        return predicted_classes, predicted_probs
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        # Return dummy predictions as fallback
        dummy_probs = np.ones((len(features_list), 4)) / 4  # Equal probabilities
        dummy_classes = np.zeros(len(features_list), dtype=int)
        return dummy_classes, dummy_probs

def fallback_classification(features):
    """使用簡單的頻段功率比較進行備用分類"""
    alpha = features.get('alpha_power', 0)
    beta = features.get('beta_power', 0)
    high_beta = features.get('high_beta_power', 0)
    theta = features.get('theta_power', 0)

    # 計算標準化波段功率
    powers = [alpha, beta, high_beta, theta]
    total = sum(powers)
    if total > 0:
        normalized = [p/total for p in powers]
    else:
        normalized = [0.25, 0.25, 0.25, 0.25]

    # 設置各個狀態的基準閾值
    alpha_threshold = 0.35  # 放鬆狀態需要較高的alpha功率
    beta_threshold = 0.30   # 專注狀態需要較高的beta功率
    high_beta_threshold = 0.30  # 壓力狀態需要較高的high_beta功率
    theta_threshold = 0.30  # 記憶狀態需要較高的theta功率

    # 檢查各頻帶是否超過閾值
    if normalized[0] > alpha_threshold and normalized[0] > normalized[1] and normalized[0] > normalized[2]:
        return 0, normalized  # 放鬆
    elif normalized[1] > beta_threshold and normalized[1] > normalized[0] and normalized[1] > normalized[3]:
        return 1, normalized  # 專注
    elif normalized[2] > high_beta_threshold and normalized[2] > normalized[0]:
        return 2, normalized  # 壓力
    elif normalized[3] > theta_threshold and normalized[3] > normalized[1]:
        return 3, normalized  # 記憶

    # 如果沒有明顯的主導頻段，選擇功率最高的
    max_idx = np.argmax(normalized)
    return max_idx, normalized

# 從use_ensemble_model.txt直接採用模型加載方法
def load_pytorch_ensemble_model(path):
    """
    Load a PyTorch ensemble model from a .pt file with improved error handling
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        # Try loading the model with custom approach
        logger.info(f"Loading model from: {path}")
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        logger.info(f"Loaded model type: {type(checkpoint)}")

        # If it's a dict with model_version, it's the new format
        if isinstance(checkpoint, dict) and 'model_version' in checkpoint:
            logger.info(f"Loading model version: {checkpoint['model_version']}")

            # Get feature names
            feature_names = checkpoint.get('feature_names', [])
            logger.info(f"Loaded {len(feature_names)} feature names")

            # Recreate base models from state dicts
            base_models = []
            for i, base_state_dict in enumerate(checkpoint['base_models_state_dict']):
                # Determine input dimension
                for key in base_state_dict:
                    if 'fc1.weight' in key:
                        input_dim = base_state_dict[key].shape[1]
                        break
                else:
                    # Default if not found
                    input_dim = 20  # Default for base models

                # Create new base model and load state dict
                logger.info(f"Creating base model {i+1} with input_dim={input_dim}")
                base_model = EEGFeatureClassifier(input_dim, checkpoint['num_classes'])
                try:
                    base_model.load_state_dict(base_state_dict)
                    base_models.append(base_model)
                    logger.info(f"Successfully loaded base model {i+1}")
                except Exception as e:
                    logger.error(f"Error loading base model {i+1}: {e}")
                    # Skip this model

            # If no base models were loaded successfully, create a default one
            if len(base_models) == 0:
                logger.warning("No base models loaded successfully, creating a default model")
                base_models.append(EEGFeatureClassifier(len(feature_names) if feature_names else 36, checkpoint.get('num_classes', 4)))

            # Recreate meta-learner
            meta_input_dim = checkpoint['num_classes'] * len(base_models)
            logger.info(f"Creating meta learner with input_dim={meta_input_dim}")
            meta_learner = MetaLearner(meta_input_dim, checkpoint['num_classes'])
            try:
                meta_learner.load_state_dict(checkpoint['meta_learner_state_dict'])
                logger.info("Successfully loaded meta learner")
            except Exception as e:
                logger.error(f"Error loading meta learner: {e}")

            # Set temperature if available
            if 'temperature' in checkpoint:
                meta_learner.temperature.data = torch.tensor([checkpoint['temperature']])
                logger.info(f"Set temperature to {checkpoint['temperature']}")

            # Create ensemble model
            selectors = checkpoint.get('selectors', [None] * len(base_models))
            scalers = checkpoint.get('scalers', [None] * len(base_models))

            ensemble_model = PyTorchEnsembleModel(
                base_models,
                selectors,
                scalers,
                meta_learner,
                checkpoint['num_classes']
            )

            scaler = None if not scalers else scalers[0]
            selector = None if not selectors else selectors[0]

            return ensemble_model, scaler, selector, feature_names

        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # This is likely just a state dict of the model
            logger.info("Found state dict. Reconstructing model...")

            # Get feature names
            feature_names = checkpoint.get('feature_names', [])
            if not feature_names:
                feature_names = [
                    'mean', 'std', 'kurtosis', 'skewness', 'activity', 'mobility', 'complexity',
                    'p2p_amp', 'rms', 'abs_mean_seg1', 'abs_mean_seg2', 'abs_mean_seg3', 'abs_mean_seg4',
                    'zero_cross_rate', 'delta_power', 'theta_power', 'alpha_power', 'beta_power',
                    'high_beta_power', 'gamma_power', 'delta_power_norm', 'theta_power_norm',
                    'alpha_power_norm', 'beta_power_norm', 'high_beta_power_norm', 'gamma_power_norm',
                    'alpha_beta_ratio', 'theta_beta_ratio', 'delta_beta_ratio', 'theta_alpha_ratio',
                    'delta_alpha_ratio', 'mad', '1st_moment', '2nd_moment', '3rd_moment', '4th_moment'
                ]

            # Create a simple model
            input_dim = len(feature_names)
            num_classes = checkpoint.get('num_classes', 4)
            model = EEGFeatureClassifier(input_dim, num_classes)

            try:
                model.load_state_dict(checkpoint['state_dict'])
                logger.info("Successfully loaded state dict into model")
            except Exception as e:
                logger.error(f"Error loading state dict: {e}")

            return model, checkpoint.get('scaler'), checkpoint.get('selector'), feature_names

        else:
            # Old format or unknown - create a default model
            logger.warning("Unknown model format, creating a default model")
            default_feature_names = [
                'mean', 'std', 'kurtosis', 'skewness', 'activity', 'mobility', 'complexity',
                'p2p_amp', 'rms', 'abs_mean_seg1', 'abs_mean_seg2', 'abs_mean_seg3', 'abs_mean_seg4',
                'zero_cross_rate', 'delta_power', 'theta_power', 'alpha_power', 'beta_power',
                'high_beta_power', 'gamma_power', 'delta_power_norm', 'theta_power_norm',
                'alpha_power_norm', 'beta_power_norm', 'high_beta_power_norm', 'gamma_power_norm',
                'alpha_beta_ratio', 'theta_beta_ratio', 'delta_beta_ratio', 'theta_alpha_ratio',
                'delta_alpha_ratio', 'mad', '1st_moment', '2nd_moment', '3rd_moment', '4th_moment'
            ]

            # Create a simple model
            model = EEGFeatureClassifier(len(default_feature_names), 4)
            return model, None, None, default_feature_names

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Create a default model
        default_feature_names = [
            'mean', 'std', 'kurtosis', 'skewness', 'activity', 'mobility', 'complexity',
            'p2p_amp', 'rms', 'abs_mean_seg1', 'abs_mean_seg2', 'abs_mean_seg3', 'abs_mean_seg4',
            'zero_cross_rate', 'delta_power', 'theta_power', 'alpha_power', 'beta_power',
            'high_beta_power', 'gamma_power', 'delta_power_norm', 'theta_power_norm',
            'alpha_power_norm', 'beta_power_norm', 'high_beta_power_norm', 'gamma_power_norm',
            'alpha_beta_ratio', 'theta_beta_ratio', 'delta_beta_ratio', 'theta_alpha_ratio',
            'delta_alpha_ratio', 'mad', '1st_moment', '2nd_moment', '3rd_moment', '4th_moment'
        ]

        model = EEGFeatureClassifier(len(default_feature_names), 4)
        return model, None, None, default_feature_names

# 實時腦電圖處理器類
import pyautogui

class RealTimeEEGProcessor:
    def __init__(self, model, scaler=None, selector=None, feature_names=None, buffer_size=500, step_size=100, fs=500):
        self.model = model
        self.scaler = scaler
        self.selector = selector
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.fs = fs
        self.buffer = collections.deque(maxlen=buffer_size)
        self.current_state = None
        self.previous_state = None
        self.state_counter = 0
        self.state_threshold = 3
        self.probabilities = [0, 0, 0, 0]
        self.start_time = time.time()
        self.command_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.transitions = 0
        
        # 新增紀錄EEG預測和手動控制的次數
        self.N_EEG = 0
        self.N_manual = 0
        
        # 新增追蹤上次手動操作的狀態
        self.last_manual_action = False  # 上次是否進行了手動操作
        self.last_manual_direction = None  # 上次手動操作的方向
        self.last_manual_steps = 0  # 上次手動操作的步數

        # 使用默認特徵名稱如果未提供
        if feature_names is None or len(feature_names) == 0:
            logger.warning("No feature names provided, using default feature names")
            self.feature_names = [
                'mean', 'std', 'kurtosis', 'skewness', 'activity', 'mobility', 'complexity',
                'p2p_amp', 'rms', 'abs_mean_seg1', 'abs_mean_seg2', 'abs_mean_seg3', 'abs_mean_seg4',
                'zero_cross_rate', 'delta_power', 'theta_power', 'alpha_power', 'beta_power',
                'high_beta_power', 'gamma_power', 'delta_power_norm', 'theta_power_norm',
                'alpha_power_norm', 'beta_power_norm', 'high_beta_power_norm', 'gamma_power_norm',
                'alpha_beta_ratio', 'theta_beta_ratio', 'delta_beta_ratio', 'theta_alpha_ratio',
                'delta_alpha_ratio', 'mad', '1st_moment', '2nd_moment', '3rd_moment', '4th_moment'
            ]
        else:
            self.feature_names = feature_names

        logger.info(f"EEG處理器初始化完成，使用 {len(self.feature_names)} 個特徵")

        # 創建日誌文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'bci_controller_{timestamp}.csv'
        self.log_file = open(log_filename, 'w')
        self.log_file.write('timestamp,previous_state,predicted_state,confidence,action,N_EEG,N_manual\n')


    def add_sample(self, sample):
        self.buffer.append(sample)

    def add_samples(self, samples):
        for sample in samples:
            self.buffer.append(sample)

    def process_buffer(self):
        """處理緩衝區中的數據並進行分類預測"""
        if len(self.buffer) < self.buffer_size:
            logger.warning(f"緩衝區中的樣本數不足，需要 {self.buffer_size}，實際有 {len(self.buffer)}")
            return None, [0, 0, 0, 0]

        # 將緩衝區數據轉換為數組
        data = list(self.buffer)
        data_array = np.array(data)

        # 記錄緩衝區統計信息
        #logger.info(f"處理數據緩衝區: 大小={len(data_array)}, 平均值={np.mean(data_array):.2f}, 範圍=[{np.min(data_array):.2f}, {np.max(data_array):.2f}]")

        # 提取特徵 - 使用from use_ensemble_model.txt的函數
        features = extract_features(data_array, self.fs)

        # 創建特徵向量（與訓練時一致的格式）
        feature_vector = np.zeros((1, len(self.feature_names)))
        for i, name in enumerate(self.feature_names):
            if name in features:
                feature_vector[0, i] = features[name]
            else:
                logger.warning(f"特徵 '{name}' 不存在於提取的特徵中")

        # 記錄關鍵波段功率
        #logger.info(f"提取的關鍵特徵: alpha_power={features.get('alpha_power', 0):.4f}, "
                  #f"beta_power={features.get('beta_power', 0):.4f}, "
                  #f"theta_power={features.get('theta_power', 0):.4f}")

        # 如果模型是PyTorchEnsembleModel，直接使用原始特徵向量進行預測
        try:
            # 確保模型處於評估模式
            self.model.eval()

            with torch.no_grad():
                # 轉換為PyTorch張量
                X_tensor = torch.FloatTensor(feature_vector)

                # 直接使用模型進行預測
                outputs = self.model(X_tensor)

                # 確保獲取概率
                if isinstance(outputs, torch.Tensor):
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    probs_numpy = probs.detach().numpy()
                else:
                    probs_numpy = np.array(outputs)

                # 如果形狀正確 (1, 4)
                if probs_numpy.shape == (1, 4):
                    probs = probs_numpy[0]
                    predicted_class = np.argmax(probs)

                    # 檢測異常情況
                    if np.sum(probs) < 0.001 or np.max(probs) < 0.1:
                        logger.warning("模型輸出異常（總和接近零或最大值太小），使用備用分類")
                        predicted_class, alt_probs = fallback_classification(features)
                        probs = np.array(alt_probs)
                        #logger.info(f"備用分類結果: 類別 {predicted_class}, 概率: {probs}")

                    # 記錄各類別概率
                    for i, p in enumerate(probs):
                        state_name = CLASS_MAPPING.get(i, f"狀態 {i}")
                        #logger.info(f"  - {state_name}: {p:.4f}")

                    # 輸出當前分類結果
                    #predicted_state_name = CLASS_MAPPING.get(predicted_class, "Unknown")
                    #logger.info(f"當前腦波分類結果: {predicted_state_name} (狀態 {predicted_class})")

                    # 更新狀態
                    self.update_state(predicted_class, probs)

                    return self.current_state, probs
                else:
                    logger.error(f"模型輸出形狀不正確: {probs_numpy.shape}")
                    predicted_class, alt_probs = fallback_classification(features)
                    self.update_state(predicted_class, alt_probs)
                    return self.current_state, alt_probs

        except Exception as e:
            logger.error(f"預測過程中出錯: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # 使用備用分類
            predicted_class, probs = fallback_classification(features)
            logger.info(f"由於錯誤使用備用分類: 類別 {predicted_class}")

            # 更新狀態
            self.update_state(predicted_class, probs)

            return self.current_state, probs

    def update_state(self, new_state, probabilities):
        """更新當前狀態並觸發相應的動作"""
        self.probabilities = probabilities
        max_prob = np.max(probabilities)

        timestamp = time.time() - self.start_time
        previous_state_name = CLASS_MAPPING.get(self.previous_state, "None")
        predicted_state_name = CLASS_MAPPING.get(new_state, "None")

        self.log_file.write(f'{timestamp:.3f},{previous_state_name},{predicted_state_name},{max_prob:.4f},')

        # 顯示當前的計數狀態
        print(f"目前統計: N_EEG={self.N_EEG}, N_manual={self.N_manual}")

        if max_prob < 0.4:
            self.log_file.write(f'none_low_confidence,{self.N_EEG},{self.N_manual}\n')
            return

        # 每次偵測到狀態時觸發EEG預測
        if new_state in STATE_TO_ACTION:
            key_name = STATE_TO_ACTION[new_state]
            logger.info(f"EEG預測: {CLASS_MAPPING[new_state]} -> 決定方向為 {key_name}")

            # 觸發EEG預測動作
            self.trigger_action(new_state, is_eeg_prediction=True)
        else:
            self.log_file.write(f'no_action_mapped,{self.N_EEG},{self.N_manual}\n')

        if new_state == self.previous_state:
            self.state_counter += 1
        else:
            self.state_counter = 1
            self.previous_state = new_state

        if self.state_counter >= self.state_threshold:
            if self.current_state != new_state:
                self.transitions += 1
                self.current_state = new_state
                self.log_file.write(f'state_changed,{self.N_EEG},{self.N_manual}\n')
            else:
                self.log_file.write(f'no_change,{self.N_EEG},{self.N_manual}\n')
        else:
            self.log_file.write(f'below_threshold,{self.N_EEG},{self.N_manual}\n')

        self.log_file.flush()

    def ask_user_for_press_count(self, key_name, is_eeg_prediction=False):
        """詢問用戶要移動幾步"""
        prompt_text = f"使用{('EEG預測' if is_eeg_prediction else '手動按鍵')}決定了方向為 {key_name}。要移動幾步？請輸入一個正整數(至少1步): "
        while True:
            try:
                press_count = int(input(prompt_text))
                if press_count >= 1:  # 確保至少移動1步
                    # 總是顯示當前的 N_manual 和 N_EEG
                    print(f"當前統計: N_manual={self.N_manual}, N_EEG={self.N_EEG}")
                    return press_count
                else:
                    print("必須至少移動1步。請輸入一個大於或等於1的正整數。")
            except ValueError:
                print("請輸入一個有效的整數。")

    def trigger_action(self, state, is_eeg_prediction=True):
        """根據狀態觸發動作，並根據規則要求手動輸入步數"""
        # 檢查上次操作是否是一步
        if self.last_manual_steps == 1:
            print(f"上次操作移動了1步，先進行手動方向和步數決定")
            
            # 詢問用戶想要手動移動的方向
            print("請選擇手動移動方向:")
            print("0. 跳過手動操作")
            print("1. 向上 (up)")
            print("2. 向下 (down)")
            print("3. 向左 (left)")
            print("4. 向右 (right)")
            
            # 获取用户选择
            choice = -1
            while choice < 0 or choice > 4:
                try:
                    choice = int(input("請輸入選擇 (0-4): "))
                    if choice < 0 or choice > 4:
                        print("無效選擇，請輸入0-4之間的數字")
                except ValueError:
                    print("請輸入一個有效的數字")
                    choice = -1
            
            # 如果选择是0，跳过手动操作
            if choice == 0:
                print("已選擇跳過手動操作，直接進行EEG預測流程")
            else:
                # 將選擇轉換為狀態
                if choice == 1:
                    manual_state = 1  # 上 - 對應狀態1
                elif choice == 2:
                    manual_state = 0  # 下 - 對應狀態0
                elif choice == 3:
                    manual_state = 2  # 左 - 對應狀態2
                elif choice == 4:
                    manual_state = 3  # 右 - 對應狀態3
                
                manual_key_name = STATE_TO_ACTION[manual_state]
                
                # 詢問用戶要手動移動幾步
                manual_steps = 0
                while manual_steps <= 0:
                    try:
                        manual_steps = int(input(f"要手動移動 {manual_key_name} 幾步？請輸入一個正整數: "))
                        if manual_steps <= 0:
                            print("請輸入一個正整數")
                    except ValueError:
                        print("請輸入一個有效的整數")
                
                # 更新手動步數計數
                self.N_manual += manual_steps
                self.command_counts[manual_state] += manual_steps
                print(f"手動控制移動總步數更新: N_manual={self.N_manual}")
                
                # 自動觸發 Google Chrome 視窗
                self.activate_chrome_window()
                
                # 執行手動動作，改用更可靠的方式
                key_mapping = {
                    "left": Key.left,
                    "down": Key.down,
                    "up": Key.up,
                    "right": Key.right
                }
                
                if manual_key_name in key_mapping:
                    key = key_mapping[manual_key_name]
                    print(f"正在執行手動移動 {manual_key_name} {manual_steps}步...")
                    
                    # 等待一下確保窗口激活
                    time.sleep(0.5)
                    
                    for i in range(manual_steps):
                        # 按下並釋放按鍵，使用0.07秒作為按鍵時間
                        keyboard_controller.press(key)
                        time.sleep(0.07)  # 按住時間精確為0.07秒
                        keyboard_controller.release(key)
                        time.sleep(0.05)  # 釋放後等待一下
                        
                        # 每按一次報告一次
                        print(f"已執行 {i+1}/{manual_steps} 步")
                    
                    print(f"已完成手動移動 {manual_key_name} {manual_steps}步")
                    
                    # 記錄到日誌
                    timestamp = time.time() - self.start_time
                    self.log_file.write(f'{timestamp:.3f},手動輸入,手動輸入,N/A,手動移動{manual_key_name}_{manual_steps}步,{self.N_EEG},{self.N_manual}\n')
                    self.log_file.flush()
        
        # 繼續原來的EEG預測流程
        if state in STATE_TO_ACTION:
            key_name = STATE_TO_ACTION[state]
            action_type = "EEG預測" if is_eeg_prediction else "手動按鍵"
            logger.info(f"{action_type}觸發動作: {CLASS_MAPPING[state]} -> 決定方向為 {key_name}")

            # 在詢問步數前先顯示當前的 N_EEG 和 N_manual
            print(f"當前統計: N_EEG={self.N_EEG}, N_manual={self.N_manual}")
            
            # 詢問用戶要移動幾步 (至少1步)
            press_count = self.ask_user_for_press_count(key_name, is_eeg_prediction)
            
            # 更新統計計數
            if is_eeg_prediction:
                self.N_EEG += press_count
                logger.info(f"EEG預測移動總步數更新: {self.N_EEG}")
            else:
                self.N_manual += press_count
                logger.info(f"手動控制移動總步數更新: {self.N_manual}")
            
            # 記錄本次操作步數，用於下次判斷
            self.last_manual_steps = press_count
            
            self.command_counts[state] += press_count
            
            # 自動觸發 Google Chrome 視窗
            self.activate_chrome_window()
            
            # 使用 pynput.keyboard 模擬按鍵，每步暫停0.07秒
            key_mapping = {
                "left": Key.left,
                "down": Key.down,
                "up": Key.up,
                "right": Key.right
            }
            
            if key_name in key_mapping:
                key = key_mapping[key_name]
                
                # 等待一下確保窗口激活
                time.sleep(0.5)
                
                for _ in range(press_count):
                    keyboard_controller.press(key)
                    # 按一個步驟停止0.07秒
                    time.sleep(0.07)
                    keyboard_controller.release(key)
                    # 每個按鍵之間暫停一下
                    time.sleep(0.05)
                
                # 再次顯示更新後的計數
                print(f"操作完成後統計: N_EEG={self.N_EEG}, N_manual={self.N_manual}")
                
                # 記錄到日誌
                timestamp = time.time() - self.start_time
                self.log_file.write(f'{timestamp:.3f},{CLASS_MAPPING.get(self.previous_state, "None")},{CLASS_MAPPING.get(state, "None")},N/A,移動{key_name}_{press_count}步,{self.N_EEG},{self.N_manual}\n')
                self.log_file.flush()
                
                return True
            else:
                logger.warning(f"鍵 {key_name} 沒有映射到 pynput.keyboard")
                return False
        else:
            logger.warning(f"狀態 {state} 沒有映射到動作")
            return False

    def activate_chrome_window(self):
        """自動觸發 Google Chrome 視窗"""
        try:
            # 使用 pyautogui 獲取屏幕尺寸
            screen_width, screen_height = pyautogui.size()

            # 假設 Google Chrome 視窗在屏幕的右側
            chrome_window_position = (screen_width // 2, screen_height // 2)

            # 移動鼠標到 Google Chrome 視窗的位置
            pyautogui.moveTo(chrome_window_position[0], chrome_window_position[1])

            # 點擊鼠標以觸發 Google Chrome 視窗
            pyautogui.click()

            #logger.info("已自動觸發 Google Chrome 視窗")
        except Exception as e:
            logger.error(f"自動觸發 Google Chrome 視窗時出錯: {e}")

    def get_performance_stats(self):
        elapsed_time = time.time() - self.start_time
        total_commands = sum(self.command_counts.values())

        stats = {
            'elapsed_time': elapsed_time,
            'total_commands': total_commands,
            'commands_per_minute': (total_commands / elapsed_time) * 60 if elapsed_time > 0 else 0,
            'state_transitions': self.transitions,
            'command_counts': self.command_counts,
            'command_distribution': {
                CLASS_MAPPING[state]: count for state, count in self.command_counts.items()
            },
            'N_EEG': self.N_EEG,
            'N_manual': self.N_manual
        }

        return stats

    def reset_performance_stats(self):
        self.command_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.start_time = time.time()
        self.transitions = 0

    def close(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

# 手動觸發狀態的函數（用於測試和WASD控制）
def manually_trigger_state(processor, state):
    """手動觸發特定狀態（用於測試或手動WASD控制）"""
    if state in STATE_TO_ACTION:
        logger.info(f"手動觸發狀態: {CLASS_MAPPING.get(state, f'狀態 {state}')}")
        processor.trigger_action(state, is_eeg_prediction=False)
        return True
    else:
        logger.warning(f"狀態 {state} 沒有映射到動作")
        return False

# 新增WASD按鍵控制函數
def setup_wasd_controls(processor):
    """設置WASD控制，用於手動方向控制"""
    def on_key_press(key):
        try:
            # 檢查按下的是否是字母鍵
            if hasattr(key, 'char'):
                if key.char.lower() == 'w':
                    logger.info("WASD控制: 向上 (W)")
                    manually_trigger_state(processor, 1)  # 上 - 對應狀態1
                elif key.char.lower() == 'a':
                    logger.info("WASD控制: 向左 (A)")
                    manually_trigger_state(processor, 2)  # 左 - 對應狀態2
                elif key.char.lower() == 's':
                    logger.info("WASD控制: 向下 (S)")
                    manually_trigger_state(processor, 0)  # 下 - 對應狀態0
                elif key.char.lower() == 'd':
                    logger.info("WASD控制: 向右 (D)")
                    manually_trigger_state(processor, 3)  # 右 - 對應狀態3
                elif key.char.lower() == 'q':
                    logger.info("按下Q鍵，準備退出...")
                    return False  # 停止監聽
        except AttributeError:
            pass
        return True

    # 創建並啟動監聽器
    from pynput.keyboard import Listener
    keyboard_listener = Listener(on_press=on_key_press)
    keyboard_listener.start()
    logger.info("WASD控制已啟用: 使用W(上), A(左), S(下), D(右) 手動控制方向")
    return keyboard_listener




# 從文件模擬腦電圖數據輸入的類（用於測試）
class EEGFileSimulator:
    def __init__(self, file_path, fs=500, speed_factor=1.0):
        self.file_path = file_path
        self.fs = fs
        self.speed_factor = speed_factor
        self.data = None
        self.index = 0
        self.running = False
        self.sample_interval = 1.0 / (fs * speed_factor)

        self.load_data()

    def load_data(self):
        try:
            if self.file_path.endswith('.txt') or self.file_path.endswith('.csv'):
                # 使用numpy的skiprows參數跳過第一行
                self.data = np.loadtxt(self.file_path, skiprows=1)
                logger.info(f"從 {self.file_path} 加載了 {len(self.data)} 個樣本 (跳過第一行)")
            else:
                logger.warning(f"不支持的文件格式: {self.file_path}")
                self.data = None
        except Exception as e:
            logger.error(f"加載數據文件時出錯: {e}")
            self.data = None

    def start(self, processor):
        if self.data is None or len(self.data) == 0:
            logger.error("沒有可用於模擬的數據")
            return

        logger.info(f"使用 {len(self.data)} 個樣本開始模擬，頻率為 {self.fs * self.speed_factor} Hz")
        self.running = True

        self.thread = threading.Thread(target=self._run_simulation, args=(processor,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

    def _run_simulation(self, processor):
        self.index = 0

        while self.running and self.index < len(self.data):
            sample = self.data[self.index]
            processor.add_sample(sample)
            time.sleep(self.sample_interval)
            self.index += 1

            if self.index % processor.step_size == 0:
                processor.process_buffer()

        logger.info("模擬完成，將重新開始模擬")
        
        # 模擬完成後重新開始，而不是停止
        if self.running:
            self._run_simulation(processor)  # 遞迴調用，重新開始模擬
        else:
            logger.info("手動停止模擬")
            

# 通過網絡接收腦電圖數據的類
class EEGNetworkReceiver:
    def __init__(self, host='0.0.0.0', port=5555, buffer_size=4096, processor=None):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.processor = processor
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.thread = None

        self.packets_received = 0
        self.samples_received = 0
        self.start_time = None
        self.last_report_time = None

    def start(self):
        if self.running:
            logger.warning("接收器已在運行")
            return

        self.running = True
        self.start_time = time.time()
        self.last_report_time = self.start_time

        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()

        logger.info(f"腦電圖網絡接收器在 {self.host}:{self.port} 上啟動")

    def stop(self):
        if not self.running:
            logger.warning("接收器未運行")
            return

        self.running = False

        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        if self.thread:
            self.thread.join(timeout=1.0)

        logger.info("腦電圖網絡接收器已停止")

    def _run_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            while self.running:
                logger.info("等待連接...")

                self.client_socket, self.client_address = self.server_socket.accept()
                logger.info(f"已連接到 {self.client_address}")

                self.client_socket.settimeout(1.0)

                self._receive_data()

                self.client_socket.close()
                self.client_socket = None

        except Exception as e:
            logger.error(f"服務器錯誤: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def _receive_data(self):
        data_buffer = b''

        while self.running and self.client_socket:
            try:
                chunk = self.client_socket.recv(self.buffer_size)

                if not chunk:
                    logger.info("客戶端斷開連接")
                    break

                data_buffer += chunk

                lines = data_buffer.split(b'\n')
                data_buffer = lines[-1]

                for line in lines[:-1]:
                    if line:
                        self._process_line(line)

                current_time = time.time()
                if current_time - self.last_report_time >= 5.0:
                    self._report_stats()
                    self.last_report_time = current_time

            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"接收數據時出錯: {e}")
                break

    def _process_line(self, line):
        try:
            line_str = line.decode('utf-8').strip()
            self.packets_received += 1

            if line_str.startswith('{') and line_str.endswith('}'):
                try:
                    data = json.loads(line_str)
                    if 'eeg' in data:
                        eeg_data = data['eeg']
                        self.samples_received += len(eeg_data)

                        if self.processor:
                            for sample in eeg_data:
                                self.processor.add_sample(sample)

                            if len(self.processor.buffer) >= self.processor.buffer_size:
                                self.processor.process_buffer()
                except json.JSONDecodeError:
                    logger.warning(f"無效的JSON數據: {line_str[:50]}...")
            else:
                try:
                    values = [float(x) for x in line_str.split(',')]
                    self.samples_received += len(values)

                    if self.processor:
                        for sample in values:
                            self.processor.add_sample(sample)

                        if len(self.processor.buffer) >= self.processor.buffer_size:
                            self.processor.process_buffer()
                except ValueError:
                    logger.warning(f"無效的CSV數據: {line_str[:50]}...")

        except Exception as e:
            logger.error(f"處理行時出錯: {e}")

    def _report_stats(self):
        if self.start_time is None:
            return

        duration = time.time() - self.start_time

        if duration > 0:
            packets_per_second = self.packets_received / duration
            samples_per_second = self.samples_received / duration

            logger.info(f"EEG接收器統計: "
                      f"{self.packets_received} 個數據包, "
                      f"{self.samples_received} 個樣本, "
                      f"{packets_per_second:.2f} 數據包/秒, "
                      f"{samples_per_second:.2f} 樣本/秒")

# 生成模擬腦電圖數據的函數
def generate_eeg_data(duration=60, fs=500):
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    base_signal = np.random.randn(n_samples)

    for i in range(1, len(base_signal)):
        base_signal[i] = 0.95 * base_signal[i-1] + 0.05 * base_signal[i]

    section_length = n_samples // 4

    relax_section = base_signal[:section_length] + 10.0 * np.sin(2 * np.pi * 10 * t[:section_length])
    concentrate_section = base_signal[section_length:2*section_length] + 8.0 * np.sin(2 * np.pi * 20 * t[section_length:2*section_length])
    stress_section = base_signal[2*section_length:3*section_length] + 6.0 * np.sin(2 * np.pi * 25 * t[2*section_length:3*section_length])
    memory_section = base_signal[3*section_length:] + 12.0 * np.sin(2 * np.pi * 6 * t[3*section_length:])

    eeg_signal = np.concatenate([relax_section, concentrate_section, stress_section, memory_section])
    eeg_signal *= 50

    logger.info(f"生成了 {len(eeg_signal)} 個模擬腦電圖數據樣本")

    return eeg_signal

# 運行腦電圖處理的主函數 - 使用新的模型加載方法
def run_eeg_processor(model_path, input_mode='simulate', eeg_file=None, simulate_duration=60, simulation_speed=1.0, network_port=5555, enable_wasd=True):
    start_time = time.time()

    # 使用來自use_ensemble_model.txt的模型加載方法
    logger.info(f"開始從 {model_path} 加載模型...")
    try:
        # 使用load_pytorch_ensemble_model加載模型
        model, scaler, selector, feature_names = load_pytorch_ensemble_model(model_path)
        logger.info("模型加載成功")

        # 設置模型為評估模式
        model.eval()
        logger.info("模型設置為評估模式")

        # 創建實時處理器
        processor = RealTimeEEGProcessor(
            model=model,
            scaler=scaler,
            selector=selector,
            feature_names=feature_names,
            buffer_size=BUFFER_SIZE,
            step_size=STEP_SIZE,
            fs=SAMPLING_RATE
        )

        # 啟用WASD控制
        keyboard_listener = None
        if enable_wasd:
            keyboard_listener = setup_wasd_controls(processor)
            logger.info("WASD控制已啟用")

        # 根據選擇的輸入模式處理
        if input_mode == 'file' and eeg_file:
            logger.info(f"從文件使用腦電圖數據: {eeg_file}")
            simulator = EEGFileSimulator(eeg_file, fs=SAMPLING_RATE, speed_factor=simulation_speed)
            simulator.start(processor)

            logger.info("按'Q'停止模擬。")
            try:
                while simulator.running:
                    time.sleep(0.5)
                    if keyboard.is_pressed('q'):  # 檢查是否按下了Q鍵
                        logger.info("檢測到Q鍵，停止模擬")
                        simulator.stop()
                        break
            except KeyboardInterrupt:
                logger.info("用户中斷")
            finally:
                simulator.stop()

        elif input_mode == 'network':
            logger.info(f"在端口 {network_port} 上啟動網絡接收器")
            receiver = EEGNetworkReceiver(port=network_port, processor=processor)
            receiver.start()

            logger.info("按'Q'停止網絡接收器。")
            try:
                while True:
                    time.sleep(0.5)
                    if keyboard.is_pressed('q'):  # 檢查是否按下了Q鍵
                        logger.info("檢測到Q鍵，停止網絡接收")
                        receiver.stop()
                        break
            except KeyboardInterrupt:
                logger.info("用户中斷")
            finally:
                receiver.stop()

        else:  # simulate 模式
            logger.info(f"使用 {simulate_duration} 秒的模擬腦電圖數據")

            simulated_data = generate_eeg_data(duration=simulate_duration, fs=SAMPLING_RATE)

            sim_file_path = f'simulated_eeg_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            np.savetxt(sim_file_path, simulated_data)
            logger.info(f"已保存模擬數據到 {sim_file_path}")

            simulator = EEGFileSimulator(sim_file_path, fs=SAMPLING_RATE, speed_factor=simulation_speed)
            simulator.start(processor)

            logger.info("按'Q'停止模擬。")
            try:
                while simulator.running:
                    time.sleep(0.5)
                    if keyboard.is_pressed('q'):  # 檢查是否按下了Q鍵
                        logger.info("檢測到Q鍵，停止模擬")
                        simulator.stop()
                        break
            except KeyboardInterrupt:
                logger.info("用户中斷")
            finally:
                simulator.stop()

        # 停止鍵盤監聽器
        if keyboard_listener:
            keyboard_listener.stop()

        # 顯示性能統計
        logger.info("處理完成。檢查性能統計...")
        stats = processor.get_performance_stats()
        logger.info(f"統計數據: 運行時間={stats['elapsed_time']:.2f}秒, 命令總數={stats['total_commands']}, 每分鐘命令數={stats['commands_per_minute']:.2f}")
        logger.info(f"EEG預測移動總步數: {stats['N_EEG']}, 手動控制移動總步數: {stats['N_manual']}")

        # 關閉處理器
        try:
            processor.close()
        except:
            pass

    except Exception as e:
        logger.error(f"運行處理器時出錯: {e}")
        import traceback
        logger.error(traceback.format_exc())

    total_time = time.time() - start_time
    logger.info(f"總運行時間: {total_time:.2f} 秒")


# 主入口點
if __name__ == "__main__":
    print("BCI迷宮遊戲控制器啟動...")

    model_path_input = input(f"輸入您的模型文件路徑（默認: {MODEL_PATH}）: ")
    if model_path_input.strip():
        MODEL_PATH = model_path_input

    if not os.path.exists(MODEL_PATH):
        print(f"未找到模型文件: {MODEL_PATH}")
        MODEL_PATH = input("請輸入有效的模型文件路徑: ")
        if not os.path.exists(MODEL_PATH):
            print(f"未找到模型文件。退出。")
            exit(1)

    print("\n選擇輸入模式:")
    print("1. 生成並使用模擬腦電圖數據")
    print("2. 從文件加載腦電圖數據")
    print("3. 通過網絡接收腦電圖數據")

    choice = input("輸入選擇 (1-3): ").strip()
    
    # 詢問是否啟用WASD控制
    enable_wasd = input("是否啟用WASD手動控制 (y/n, 默認: y): ").strip().lower() != 'n'

    if choice == '1':
        input_mode = 'simulate'
        eeg_file = None

        try:
            duration = int(input("輸入模擬時間（秒）（默認=60）: ") or "60")
        except ValueError:
            duration = 60

        try:
            speed = float(input("輸入模擬速度因子（1.0 = 實時，默認=1.0）: ") or "1.0")
        except ValueError:
            speed = 1.0

        run_eeg_processor(
            MODEL_PATH,
            input_mode=input_mode,
            simulate_duration=duration,
            simulation_speed=speed,
            enable_wasd=enable_wasd
        )

    elif choice == '2':
        input_mode = 'file'

        eeg_file = input("輸入腦電圖數據文件路徑: ")
        if not os.path.exists(eeg_file):
            print(f"未找到文件: {eeg_file}")
            eeg_file = input("請輸入有效的文件路徑: ")
            if not os.path.exists(eeg_file):
                print("未找到文件。退出。")
                exit(1)

        try:
            speed = float(input("輸入模擬速度因子（1.0 = 實時，默認=1.0）: ") or "1.0")
        except ValueError:
            speed = 1.0

        run_eeg_processor(
            MODEL_PATH,
            input_mode=input_mode,
            eeg_file=eeg_file,
            simulation_speed=speed,
            enable_wasd=enable_wasd
        )

    elif choice == '3':
        input_mode = 'network'

        try:
            port = int(input("輸入網絡端口（默認=5555）: ") or "5555")
        except ValueError:
            port = 5555

        run_eeg_processor(
            MODEL_PATH,
            input_mode=input_mode,
            network_port=port,
            enable_wasd=enable_wasd
        )

    else:
        print("無效選擇。使用模擬數據。")
        run_eeg_processor(MODEL_PATH, input_mode='simulate', simulate_duration=60, simulation_speed=1.0, enable_wasd=enable_wasd)