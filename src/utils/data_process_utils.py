"""
Tiện ích Xử lý Dữ liệu - Các hàm làm việc với dữ liệu

=== MỤC ĐÍCH: XỬ LÝ DỮ LIỆU ===

Bao gồm tất cả các hàm cần thiết cho:
1. Đọc và tải dữ liệu an toàn
2. Làm sạch và tiền xử lý dữ liệu  
3. Tối ưu hóa memory và performance
4. Chia batch và chunking
5. Validate và transform dữ liệu

Code đơn giản, dễ hiểu, dễ sử dụng.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import warnings
import json


# ==============================================================================
# 1. ĐỌC VÀ TẢI DỮ LIỆU
# ==============================================================================

def tai_du_lieu_chunked(file_path: str, 
                        chunk_size: int = 10000, 
                        max_rows: Optional[int] = None,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Tải dữ liệu lớn theo chunks để tiết kiệm memory
    
    Hữu ích cho file CSV rất lớn không thể load hết vào memory.
    
    Tham số:
        file_path: đường dẫn đến file
        chunk_size: số rows mỗi chunk
        max_rows: giới hạn tổng số rows (None = không giới hạn)
        columns: danh sách columns cần load (None = load tất cả)
    
    Trả về:
        DataFrame: dữ liệu đã được gộp từ các chunks
    """
    chunks = []
    total_rows = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=columns):
            # Làm sạch chunk
            chunk = lam_sach_ten_cot(chunk)
            chunk = toi_uu_memory_dataframe(chunk)
            
            chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows % 100000 == 0:
                print(f"Loaded {total_rows} rows")
            # Kiểm tra giới hạn rows
            if max_rows and total_rows >= max_rows:
                break

    except Exception as e:
        print(f"Lỗi khi đọc chunks: {e}")
        if chunks:
            print(f"Đã đọc được {len(chunks)} chunks trước khi lỗi")
        else:
            raise
    
    # Gộp tất cả chunks
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        if max_rows:
            df = df.head(max_rows)
        return df
    else:
        return pd.DataFrame()


# ==============================================================================
# 2. LÀM SẠCH VÀ TIỀN XỬ LÝ
# ==============================================================================

def lam_sach_ten_cot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch tên cột để dễ sử dụng
    
    - Loại bỏ khoảng trắng thừa
    - Chuyển về lowercase
    - Thay thế ký tự đặc biệt bằng underscore
    
    Tham số:
        df: DataFrame cần làm sạch tên cột
    
    Trả về:
        DataFrame: với tên cột đã được làm sạch
    """
    df = df.copy()
    
    # Làm sạch tên cột
    new_columns = []
    for col in df.columns:
        # Chuyển về string và strip
        new_col = str(col).strip()
        
        # Chuyển về lowercase
        new_col = new_col.lower()
        
        # Thay thế khoảng trắng và ký tự đặc biệt
        import re
        new_col = re.sub(r'[^a-zA-Z0-9_]', '_', new_col)
        
        # Loại bỏ underscore liên tiếp
        new_col = re.sub(r'_+', '_', new_col)
        
        # Loại bỏ underscore ở đầu/cuối
        new_col = new_col.strip('_')
        
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


def xu_ly_gia_tri_null(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """
    Xử lý giá trị null trong DataFrame
    
    Tham số:
        df: DataFrame cần xử lý
        strategy: chiến lược xử lý ('auto', 'drop', 'fill_mean', 'fill_median', 'fill_mode')
    
    Trả về:
        DataFrame: đã xử lý null values
    """
    df = df.copy()
    
    if strategy == 'auto':
        # Tự động chọn strategy tốt nhất cho từng cột
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            
            if null_ratio > 0.5:
                # Quá nhiều null, xóa cột
                df = df.drop(columns=[col])
            elif df[col].dtype in ['int64', 'float64']:
                # Numeric: fill median
                df[col] = df[col].fillna(df[col].median())
            else:
                # Categorical: fill mode
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
    
    elif strategy == 'drop':
        df = df.dropna()
    
    elif strategy == 'fill_mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    elif strategy == 'fill_median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    elif strategy == 'fill_mode':
        for col in df.columns:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
    
    return df


def tach_dac_trung_va_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tách DataFrame thành đặc trưng (X) và target (y)
    
    Tham số:
        df: DataFrame chứa toàn bộ dữ liệu
        target_col: tên cột target
    
    Trả về:
        X: DataFrame đặc trưng
        y: Series target
    """
    if target_col not in df.columns:
        raise ValueError(f"Cột target '{target_col}' không tồn tại trong DataFrame")
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    return X, y


def chuan_hoa_du_lieu(X: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Chuẩn hóa dữ liệu đặc trưng
    
    Tham số:
        X: DataFrame đặc trưng
        method: phương pháp chuẩn hóa ('standard', 'minmax', 'robust')
    
    Trả về:
        X_scaled: DataFrame đã chuẩn hóa
        scaler_params: tham số để inverse transform
    """
    X_scaled = X.copy()
    scaler_params = {}
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'standard':
            # Z-score normalization: (x - mean) / std
            mean_val = X[col].mean()
            std_val = X[col].std()
            if std_val > 0:
                X_scaled[col] = (X[col] - mean_val) / std_val
                scaler_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'standard'}
        
        elif method == 'minmax':
            # Min-max scaling: (x - min) / (max - min)
            min_val = X[col].min()
            max_val = X[col].max()
            if max_val > min_val:
                X_scaled[col] = (X[col] - min_val) / (max_val - min_val)
                scaler_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        elif method == 'robust':
            # Robust scaling: (x - median) / IQR
            median_val = X[col].median()
            q75 = X[col].quantile(0.75)
            q25 = X[col].quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                X_scaled[col] = (X[col] - median_val) / iqr
                scaler_params[col] = {'median': median_val, 'iqr': iqr, 'method': 'robust'}
    
    return X_scaled, scaler_params


# ==============================================================================
# 3. TỐI ƯU HÓA MEMORY VÀ PERFORMANCE
# ==============================================================================

def toi_uu_memory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tối ưu hóa memory usage của DataFrame
    
    Tự động chuyển các cột về dtype phù hợp nhất để tiết kiệm memory.
    
    Tham số:
        df: DataFrame cần tối ưu
    
    Trả về:
        DataFrame: đã được tối ưu memory
    """
    df = df.copy()
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            # Thử convert sang numeric
            numeric_converted = pd.to_numeric(df[col], errors='ignore')
            if numeric_converted.dtype != 'object':
                df[col] = numeric_converted
                col_type = df[col].dtype
        
        if col_type in ['int64', 'int32']:
            # Tối ưu integer columns
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        elif col_type in ['float64', 'float32']:
            # Tối ưu float columns
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        
        elif col_type == 'object':
            # Thử convert sang category nếu có ít unique values
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            if num_unique / num_total < 0.5:  # Nếu < 50% unique
                df[col] = df[col].astype('category')
    
    return df


def lay_thong_tin_du_lieu(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Lấy thông tin tổng quan về DataFrame
    
    Tham số:
        df: DataFrame cần phân tích
    
    Trả về:
        dict: thông tin chi tiết về dữ liệu
    """
    info = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['category', 'object']).columns),
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Thống kê cho numeric columns
    if info['numeric_columns']:
        info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
    
    return info


# ==============================================================================
# 4. CHIA BATCH VÀ CHUNKING
# ==============================================================================

def tao_batches(X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
               shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Chia dữ liệu thành các batches cho training
    
    Tham số:
        X: ma trận đặc trưng
        y: vector target
        batch_size: kích thước mỗi batch
        shuffle: có shuffle dữ liệu không
    
    Trả về:
        List[Tuple]: danh sách các (X_batch, y_batch)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        batches.append((X_batch, y_batch))
    
    return batches


def chia_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chia dữ liệu thành tập train và test
    
    Tham số:
        X: ma trận đặc trưng
        y: vector target  
        test_size: tỷ lệ test set (0.0 - 1.0)
        random_state: seed cho reproducibility
    
    Trả về:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices] 
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


# ==============================================================================
# 5. VALIDATE VÀ TRANSFORM
# ==============================================================================

def kiem_tra_du_lieu_dau_vao(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Kiểm tra tính hợp lệ của dữ liệu đầu vào
    
    Tham số:
        X: ma trận đặc trưng
        y: vector target
    
    Trả về:
        bool: True nếu dữ liệu hợp lệ
    """
    try:
        # Kiểm tra shape
        if len(X.shape) != 2:
            print("Lỗi: X phải là ma trận 2D")
            return False
        
        if len(y.shape) != 1:
            print("Lỗi: y phải là vector 1D")
            return False
        
        if X.shape[0] != y.shape[0]:
            print("Lỗi: Số samples trong X và y không khớp")
            return False
        
        # Kiểm tra giá trị
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Cảnh báo: X chứa NaN hoặc Inf")
            return False
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Cảnh báo: y chứa NaN hoặc Inf")
            return False
        
        print(f"Dữ liệu hợp lệ: {X.shape[0]} samples, {X.shape[1]} features")
        return True
        
    except Exception as e:
        print(f"Lỗi khi kiểm tra dữ liệu: {e}")
        return False


def chuyen_pandas_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Chuyển DataFrame pandas sang numpy array an toàn
    
    Tham số:
        df: DataFrame cần chuyển đổi
    
    Trả về:
        np.ndarray: mảng numpy
    """
    # Chỉ lấy numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("Không có cột nào là numeric để chuyển đổi")
    
    return numeric_df.values.astype(np.float64)


def load_du_lieu():
    """
    Load dữ liệu từ 02_processed directory (data đã được preprocessed)
    
    Trả về:
        X_train, X_test, y_train, y_test: numpy arrays
        
    Note: Target data đã được log-transformed trong preprocessing pipeline
    """
    data_dir = Path("data/02_processed")
    
    # Check if directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} không tồn tại. Hãy chạy preprocessing trước.")
    
    # Load data files
    try:
        X_train = tai_du_lieu_chunked(data_dir / "X_train.csv").values
        X_test = tai_du_lieu_chunked(data_dir / "X_test.csv").values
        y_train = tai_du_lieu_chunked(data_dir / "y_train.csv").values.ravel()
        y_test = tai_du_lieu_chunked(data_dir / "y_test.csv").values.ravel()
        
        print(f"✅ Đã load dữ liệu: Train {X_train.shape}, Test {X_test.shape}")
        print(f"   Target (log): y_train [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"   Target (log): y_test [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        # Load and display metadata if available
        feature_info_path = data_dir / "feature_info.json"
        if feature_info_path.exists():
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
            
            print(f"   📋 Số features: {feature_info['n_features']}")
            print(f"   🔄 Biến đổi: {feature_info['target_info']['transformation']}")
            print(f"   ✅ Sẵn sàng cho thuật toán")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Không tìm thấy file data: {e}. Hãy chạy preprocessing pipeline trước.")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi load data: {e}")


def in_thong_tin_du_lieu(df: pd.DataFrame, ten_dataset: str = "Dataset"):
    """
    In thông tin tổng quan về dataset
    """
    info = lay_thong_tin_du_lieu(df)
    
    print(f"\n=== {ten_dataset} ===")
    print(f"Kích thước: {info['shape']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    print(f"Duplicate rows: {info['duplicate_rows']}")
    print(f"Numeric columns: {len(info['numeric_columns'])}")
    print(f"Categorical columns: {len(info['categorical_columns'])}")
    
    # Null values
    null_cols = {k: v for k, v in info['null_counts'].items() if v > 0}
    if null_cols:
        print(f"Null values: {null_cols}")
    else:
        print("Không có null values")