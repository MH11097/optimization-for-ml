"""
Utils Package - Tiện ích cho Newton Method

Bao gồm 3 module chính với tên file tiếng Anh, tên hàm tiếng Việt:

1. optimization_utils.py - Các hàm tối ưu hóa
2. data_process_utils.py - Các hàm xử lý dữ liệu  
3. visualization_utils.py - Các hàm trực quan hóa

Tất cả hàm có tên tiếng Việt dễ hiểu, code đơn giản, dễ sử dụng.
"""

# Import các hàm chính cho dễ sử dụng
from .optimization_utils import (
    # Tính toán cơ bản
    tinh_gradient_hoi_quy_tuyen_tinh,
    tinh_ma_tran_hessian_hoi_quy_tuyen_tinh,
    giai_he_phuong_trinh_tuyen_tinh,
    
    # Kiểm tra ma trận
    kiem_tra_positive_definite,
    tinh_condition_number,
    
    # Đánh giá mô hình
    du_doan,
    tinh_mse,
    tinh_mae,
    tinh_r2_score,
    
    # Các hàm loss
    tinh_loss_ols,
    tinh_loss_ridge,
    tinh_loss_lasso_smooth,
    
    # Hội tụ và line search
    kiem_tra_hoi_tu,
    backtracking_line_search,
    
    # Debug utilities
    in_thong_tin_ma_tran,
    in_thong_tin_gradient
)

from .data_process_utils import (
    # Đọc dữ liệu
    tai_du_lieu_chunked,
    
    # Tiền xử lý
    lam_sach_ten_cot,
    xu_ly_gia_tri_null,
    tach_dac_trung_va_target,
    chuan_hoa_du_lieu,
    
    # Tối ưu hóa
    toi_uu_memory_dataframe,
    lay_thong_tin_du_lieu,
    
    # Chia dữ liệu
    tao_batches,
    chia_train_test,
    
    # Validate
    kiem_tra_du_lieu_dau_vao,
    chuyen_pandas_to_numpy,
    in_thong_tin_du_lieu
)

from .visualization_utils import (
    # Setup
    thiet_lap_style_bieu_do,
    tao_color_palette,
    
    # Training curves
    ve_duong_hoi_tu,
    ve_so_sanh_algorithms,
    
    # Phân tích predictions
    ve_du_doan_vs_thuc_te,
    ve_phan_tich_residuals,
    
    # So sánh performance
    ve_bang_so_sanh_performance,
    ve_radar_chart_algorithms,
    
    # Ma trận và gradient
    ve_ma_tran_heatmap,
    ve_gradient_vector,
    
    # Báo cáo tổng hợp
    tao_bao_cao_visual_tong_hop,
    luu_bieu_do_theo_batch
)

__all__ = [
    # Optimization
    'tinh_gradient_hoi_quy_tuyen_tinh',
    'tinh_ma_tran_hessian_hoi_quy_tuyen_tinh', 
    'giai_he_phuong_trinh_tuyen_tinh',
    'kiem_tra_positive_definite',
    'tinh_condition_number',
    'du_doan',
    'tinh_mse',
    'tinh_mae', 
    'tinh_r2_score',
    'tinh_loss_ols',
    'tinh_loss_ridge',
    'tinh_loss_lasso_smooth',
    'kiem_tra_hoi_tu',
    'backtracking_line_search',
    'in_thong_tin_ma_tran',
    'in_thong_tin_gradient',
    
    # Data processing
    'doc_csv_an_toan',
    'tai_du_lieu_chunked',
    'lam_sach_ten_cot',
    'xu_ly_gia_tri_null',
    'tach_dac_trung_va_target',
    'chuan_hoa_du_lieu',
    'toi_uu_memory_dataframe',
    'lay_thong_tin_du_lieu',
    'tao_batches',
    'chia_train_test',
    'kiem_tra_du_lieu_dau_vao',
    'chuyen_pandas_to_numpy',
    'in_thong_tin_du_lieu',
    
    # Visualization
    'thiet_lap_style_bieu_do',
    'tao_color_palette',
    've_duong_hoi_tu',
    've_so_sanh_algorithms',
    've_du_doan_vs_thuc_te',
    've_phan_tich_residuals',
    've_bang_so_sanh_performance',
    've_radar_chart_algorithms',
    've_ma_tran_heatmap',
    've_gradient_vector',
    'tao_bao_cao_visual_tong_hop',
    'luu_bieu_do_theo_batch'
]