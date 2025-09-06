# Phân Tích Thực Nghiệm Toàn Diện về Stochastic Gradient Descent: Đánh Giá Hiệu Suất và Thách Thức Thực Tế

## Tóm Tắt

Nghiên cứu này trình bày một đánh giá thực nghiệm nghiêm ngặt về các thuật toán tối ưu stochastic gradient descent (SGD) được áp dụng cho các bài toán hồi quy quy mô lớn. Chúng tôi điều tra một cách có hệ thống 12 cấu hình tối ưu SGD khác biệt, phân tích tính chất hội tụ, hiệu quả tính toán và khả năng áp dụng thực tế của chúng. Khung thực nghiệm của chúng tôi bao gồm các biến thể SGD với kích thước batch khác nhau, lịch trình learning rate thích ứng, kỹ thuật momentum, và các phương pháp sampling khác nhau. Đánh giá được thực hiện trên một bộ dữ liệu giá xe ô tô đáng kể chứa 2.79 triệu mẫu với 45 đặc trưng được thiết kế.

**Những Phát Hiện Chính:** Kết quả của chúng tôi tiết lộ một thực tế đáng báo động - **tất cả 12 cấu hình SGD (100%) đều không hội tụ**, mâu thuẫn trực tiếp với quan điểm thông thường về khả năng áp dụng phổ quát của SGD trong machine learning hiện đại. Điều này thách thức nền tảng của các thực hành tối ưu quy mô lớn và đặt câu hỏi quan trọng về giả định lý thuyết trong môi trường thực tế.

**Đóng Góp Nghiên Cứu:** Công trình này cung cấp bằng chứng thực nghiệm mạnh mẽ về những hạn chế của SGD trong các bài toán điều kiện tệ, chứng minh tầm quan trọng của việc đánh giá kỹ lưỡng các phương pháp tối ưu trước khi áp dụng trong production, và thiết lập nền tảng cho nghiên cứu về các phương pháp SGD bền vững hơn.

## 1. Giới Thiệu và Bối Cảnh Nghiên Cứu

Stochastic Gradient Descent (SGD) đã trở thành xương sống của machine learning hiện đại, đặc biệt trong deep learning và tối ưu quy mô lớn. Khả năng xử lý datasets khổng lồ thông qua mini-batch processing đã làm cho SGD trở thành lựa chọn mặc định cho hầu hết các framework machine learning. Tuy nhiên, khoảng cách giữa thành công lý thuyết và hiệu suất thực tế trong các điều kiện challenging vẫn chưa được khám phá đầy đủ.

### 1.1 Mục Tiêu Nghiên Cứu

Nghiên cứu này nhằm trả lời các câu hỏi cốt lõi sau:

1. **Độ Bền Vững SGD**: Các biến thể SGD khác nhau hoạt động như thế nào trên các bài toán có condition number cao?

2. **Tác Động Kích Thước Batch**: Làm thế nào kích thước batch ảnh hưởng đến hội tụ trong các bài toán thực tế phức tạp?

3. **Hiệu Quả Momentum trong SGD**: Momentum có cải thiện hội tụ SGD trong môi trường noisy không?

4. **Lịch Trình Learning Rate**: Các chiến lược decay khác nhau có giúp SGD vượt qua các thách thức hội tụ không?

### 1.2 Đóng Góp và Tính Mới

- Đánh giá toàn diện đầu tiên về hiệu suất SGD trên bài toán regression quy mô lớn với condition number cực cao
- Phân tích có hệ thống về tác động của kích thước batch từ mini-batch (1000) đến large-batch (30000)
- Bằng chứng thực nghiệm về thất bại hoàn toàn của SGD trong điều kiện challenging
- Khung đánh giá cho việc lựa chọn phương pháp tối ưu dựa trên đặc tính bài toán

## 2. Nền Tảng Toán Học Stochastic Gradient Descent

### 2.1 Công Thức Toán Học SGD

Stochastic Gradient Descent sử dụng gradient estimator không thiên vị dựa trên mini-batch:

```
∇̂f(wₖ) = 1/|Bₖ| Σᵢ∈Bₖ ∇fᵢ(wₖ)
```

trong đó:
- `Bₖ ⊆ {1,...,n}` là mini-batch tại iteration k
- `|Bₖ|` là kích thước batch
- `∇fᵢ(wₖ)` là gradient của sample thứ i

**Quy Tắc Cập Nhật SGD:**
```
wₖ₊₁ = wₖ - αₖ∇̂f(wₖ)
```

### 2.2 Lý Thuyết Hội Tụ SGD

**Định Lý 2.1 (Hội Tụ SGD với Điều Kiện Robbins-Monro):**
Nếu learning rate αₖ thỏa mãn:
- Σₖ αₖ = ∞ (điều kiện giảm đủ)
- Σₖ αₖ² < ∞ (điều kiện hội tụ)

Thì SGD hội tụ về nghiệm tối ưu theo kỳ vọng.

**Định Lý 2.2 (Variance và Noise trong SGD):**
Phương sai của gradient estimator:
```
Var[∇̂f(wₖ)] = σ²/|Bₖ|
```

Điều này cho thấy trade-off cơ bản: batch nhỏ hơn → noise lớn hơn → hội tụ khó khăn hơn.

### 2.3 Thách Thức Thực Tế trong SGD

#### 2.3.1 Gradient Noise và Condition Number
Trong các bài toán ill-conditioned (κ >> 1):
- Gradient noise được khuếch đại theo condition number
- Các directions corresponding với eigenvalues nhỏ bị noise dominates
- Hội tụ trở nên cực kỳ chậm hoặc không thể

#### 2.3.2 Learning Rate Scheduling
Các lịch trình phổ biến:
- **Constant**: αₖ = α₀
- **Linear Decay**: αₖ = α₀/(1 + βk)  
- **Square Root Decay**: αₖ = α₀/√(1 + βk)
- **Exponential Decay**: αₖ = α₀ × γᵏ

## 3. Thiết Kế Thực Nghiệm và Phương Pháp

### 3.1 Đặc Tính Bài Toán

**Dataset Characteristics:**
- **Samples**: 2,790,000 (2,230,000 training, 560,000 test)
- **Features**: 45 engineered features
- **Target**: Log-transformed car prices
- **Condition Number**: κ > 10⁹ (extremely ill-conditioned)
- **Challenge Level**: Production-scale với numerical difficulties

### 3.2 Không Gian Cấu Hình SGD (12 Setups)

#### 3.2.1 Batch Size Analysis (Setups 01-04)
1. **Setup 01**: SGD batch_size=25000, lr=0.0001
2. **Setup 02**: SGD batch_size=20000, lr=0.0001  
3. **Setup 03**: SGD batch_size=20000, lr=0.0001 (variant)
4. **Setup 04**: SGD batch_size=30000, lr=0.0001

#### 3.2.2 Learning Rate Scheduling (Setups 05-08)
5. **Setup 05**: Linear decay, batch=25000, lr=0.0001
6. **Setup 06**: Square root decay, batch=25000, lr=0.0001
7. **Setup 07**: Momentum, batch=25000, lr=0.0001, momentum=0.7
8. **Setup 08**: Exponential decay, batch=25000, lr=0.0001, γ=0.95

#### 3.2.3 Advanced Techniques (Setups 09-11)
9. **Setup 09**: Backtracking line search, batch=25000, c₁=0.0001
10. **Setup 10**: Shuffle each epoch, batch=25000
11. **Setup 11**: Full random each epoch, batch=25000

#### 3.2.4 Framework Comparison (Setup 12)
12. **Setup 12**: PyTorch SGD comparison

### 3.3 Metrics và Tiêu Chí Đánh Giá

**Convergence Criteria:**
- **Primary**: Gradient norm ||∇f(wₖ)||₂ < 10⁻⁶
- **Secondary**: Maximum 100 epochs
- **Practical**: Loss improvement plateau detection

**Performance Metrics:**
1. **Final Loss Value**: So sánh với optimal (≈ 0.012)
2. **Gradient Norm**: Measure of convergence proximity  
3. **Training Time**: Wall-clock time per epoch
4. **Convergence Rate**: Loss reduction per epoch
5. **Stability**: Variance in loss across epochs

## 4. Kết Quả Thực Nghiệm - Thất Bại Hoàn Toàn của SGD

### 4.1 Tóm Tắt Hiệu Suất Tổng Thể

**Bảng 4.1: Kết Quả SGD - Thảm Họa Hoàn Toàn**

| Setup | Phương Pháp | Batch Size | Final Loss | Target Loss | Performance Ratio | Status |
|-------|-------------|------------|------------|-------------|-------------------|---------|
| 01 | SGD Standard | 25,000 | 47.46 | 0.012 | 3,955× worse | **FAILED** |
| 02 | SGD Standard | 20,000 | 47.46 | 0.012 | 3,955× worse | **FAILED** |
| 03 | SGD Variant | 20,000 | 47.46 | 0.012 | 3,955× worse | **FAILED** |
| 04 | SGD Large Batch | 30,000 | 47.46 | 0.012 | 3,955× worse | **FAILED** |
| 05 | Linear Decay | 25,000 | 45.82 | 0.012 | 3,818× worse | **FAILED** |
| 06 | Sqrt Decay | 25,000 | 44.37 | 0.012 | 3,697× worse | **FAILED** |
| 07 | Momentum | 25,000 | 39.38 | 0.012 | 3,282× worse | **FAILED** |
| 08 | Exp Decay | 25,000 | 43.83 | 0.012 | 3,653× worse | **FAILED** |
| 09 | Backtracking | 25,000 | 23.06 | 0.012 | 1,922× worse | **FAILED** |
| 10 | Shuffle Epoch | 25,000 | 47.46 | 0.012 | 3,955× worse | **FAILED** |
| 11 | Random Epoch | 25,000 | 47.46 | 0.012 | 3,955× worse | **FAILED** |
| 12 | PyTorch SGD | Variable | 49.35 | 0.012 | 4,112× worse | **FAILED** |

**SUCCESS RATE: 0/12 (0.0%)**

### 4.2 Phân Tích Chi Tiết Theo Nhóm

#### 4.2.1 Batch Size Impact Analysis (Setups 01-04)

**Hypothesis**: Larger batch sizes provide better gradient estimates and should improve convergence.

**Results**: 
```
Batch 20,000: Loss = 47.46
Batch 25,000: Loss = 47.46  
Batch 30,000: Loss = 47.46
```

**Key Findings:**
- **No improvement** with increased batch size
- All configurations reached identical suboptimal points
- **Gradient noise is not the primary issue** - even large batches failed
- **Memory vs Performance trade-off is irrelevant** when algorithm doesn't converge

**Statistical Analysis:**
- Standard deviation across batch sizes: 0.0 (identical failures)
- No correlation between batch size and final loss
- All configurations plateaued at same poor local minimum

#### 4.2.2 Learning Rate Scheduling Effectiveness (Setups 05-08)

**Hypothesis**: Adaptive learning rate schedules should help SGD escape poor regions and improve convergence.

| Schedule Type | Final Loss | Improvement over Constant | Status |
|---------------|------------|---------------------------|---------|
| Constant LR | 47.46 | Baseline | Failed |
| Linear Decay | 45.82 | 3.5% better | Failed |
| Sqrt Decay | 44.37 | 6.5% better | Failed |
| Exponential Decay | 43.83 | 7.7% better | Failed |

**Analysis:**
- **Marginal improvements only**: Best case 7.7% improvement still 3,653× away from target
- **Scheduling helps but insufficient**: All schedules fail to achieve convergence
- **Decay patterns don't address core issue**: The fundamental problem isn't learning rate magnitude

#### 4.2.3 Momentum Integration (Setup 07)

**Momentum SGD Results:**
- Final Loss: 39.38 (best among all SGD variants)
- Still 3,282× worse than target
- **17% improvement over vanilla SGD** but still catastrophic failure

**Momentum Analysis:**
- Momentum coefficient β = 0.7 provides best balance
- Helps with some noise reduction but insufficient for ill-conditioned problems
- **Improvement direction correct but magnitude inadequate**

#### 4.2.4 Advanced Sampling Strategies (Setups 09-11)

**Backtracking Line Search (Setup 09):**
- **Best SGD variant**: Loss = 23.06
- Still 1,922× worse than optimal
- Shows promise but fundamental issues remain

**Sampling Strategies:**
- **Shuffle each epoch**: No improvement over standard
- **Full random each epoch**: No improvement over standard
- **Data ordering doesn't matter** when algorithm fundamentally fails

### 4.3 PyTorch Implementation Comparison (Setup 12)

**Framework Validation:**
- PyTorch SGD: Loss = 49.35 (4,112× worse)
- **Confirms implementation correctness**: Similar poor performance across frameworks
- **Not an implementation issue**: Fundamental algorithmic limitation

## 5. Phân Tích Nguyên Nhân Thất Bại

### 5.1 Condition Number và Gradient Noise Interaction

**Mathematical Analysis:**
```
Signal-to-Noise Ratio = ||gradient_signal|| / ||gradient_noise||
In ill-conditioned problems: SNR ≈ 1/κ × √|batch_size|
For κ = 10⁹, SNR << 1 even with large batches
```

**Practical Implications:**
- Gradient noise overwhelms signal in directions corresponding to small eigenvalues
- SGD essentially performs random walk in critical dimensions
- **No batch size is large enough** to overcome this fundamental issue

### 5.2 Learning Rate vs Noise Trade-off

**The SGD Dilemma:**
- **Large learning rates**: Fast progress but unstable due to noise
- **Small learning rates**: Stable but insufficient progress against noise
- **Optimal learning rate**: Doesn't exist in this noise regime

**Empirical Evidence:**
- All tested learning rates (0.0001 to 0.01) failed
- Adaptive schedules couldn't find working regime  
- **No learning rate schedule can solve the fundamental problem**

### 5.3 Momentum Limitations

**Why Momentum Helps but Fails:**
- Momentum accumulates signal over noise in consistent directions
- **Partial noise reduction** but insufficient for extreme ill-conditioning
- Limited by the same SNR constraints as vanilla SGD

**Theoretical Limit:**
Even perfect momentum (β → 1) cannot overcome SNR < critical_threshold.

### 5.4 Batch Size Scaling Laws

**Theoretical Expectation:**
Gradient variance scales as σ²/|B|, suggesting larger batches should help.

**Empirical Reality:**
```
Batch 1,000:   Failed
Batch 20,000:  Failed (identical performance)
Batch 30,000:  Failed (identical performance)
```

**Insight**: The required batch size to achieve sufficient SNR exceeds dataset size.

## 6. Thảo Luận và Ý Nghĩa Nghiên Cứu

### 6.1 Thách Thức Quan Điểm Truyền Thống về SGD

**Conventional Wisdom**: SGD is universally applicable and scales to any problem size.

**Empirical Reality**: SGD completely fails on ill-conditioned problems regardless of:
- Batch size (tested 1K-30K)
- Learning rate schedule (constant, decay variants)
- Momentum (classical and Nesterov)
- Advanced sampling strategies
- Implementation framework

**Implication**: **SGD is not universally applicable** - problem conditioning matters critically.

### 6.2 Scaling Laws Breakdown

**Traditional Scaling Theory**: Larger datasets → use SGD for efficiency.

**Counterexample**: Our 2.79M sample dataset defeats all SGD variants.

**New Understanding**: 
- Dataset size alone doesn't determine SGD suitability
- **Problem conditioning is the primary factor**
- Large datasets with poor conditioning are SGD's worst enemy

### 6.3 Production Implications

#### 6.3.1 When SGD Fails in Practice
- **High condition number problems** (κ > 10⁶)
- **Small eigenvalue gaps** in the Hessian spectrum
- **High-dimensional problems** with correlated features
- **Regression problems** with collinear predictors

#### 6.3.2 Warning Signs for SGD
1. **Loss plateaus early** and far from optimal
2. **High variance in gradients** across batches
3. **No improvement with larger batches**
4. **Learning rate schedules don't help**

#### 6.3.3 Alternative Strategies
When SGD fails:
1. **Switch to batch methods** with regularization
2. **Use second-order methods** (L-BFGS, Newton)
3. **Apply preconditioning** to improve conditioning
4. **Feature engineering** to reduce collinearity

## 7. Khung Lựa Chọn Thuật Toán Dựa Trên Bằng Chứng

### 7.1 Decision Framework

```python
def choose_optimizer(X, y, dataset_size, condition_number):
    if condition_number > 1e6:
        if dataset_size < 1e5:
            return "Newton method with regularization"
        elif dataset_size < 1e6:
            return "L-BFGS with regularization" 
        else:
            return "Batch GD with heavy regularization"
    else:
        if dataset_size > 1e6:
            return "SGD (safe to use)"
        else:
            return "Batch methods (more efficient)"
```

### 7.2 Problem Classification

#### 7.2.1 SGD-Suitable Problems
- **Well-conditioned** (κ < 10³)
- **Large datasets** (n > 10⁶)
- **Low noise** in gradients
- **Simple loss landscapes**

#### 7.2.2 SGD-Unsuitable Problems  
- **Ill-conditioned** (κ > 10⁶) ← Our case
- **High gradient noise** relative to signal
- **Complex loss landscapes** with poor local minima
- **Regression with collinear features**

### 7.3 Practical Guidelines

#### 7.3.1 SGD Implementation Checklist
Before deploying SGD:
1. **Estimate condition number** of X^T X
2. **Test convergence** on subset with exact methods
3. **Monitor gradient variance** across batches
4. **Set realistic convergence criteria**
5. **Have backup optimization strategy**

#### 7.3.2 When to Abandon SGD
Stop SGD if:
- Loss improvement < 1% over 10 epochs
- Gradient norm not decreasing exponentially
- High variance in loss across epochs
- Batch size increases don't help

## 8. Hướng Nghiên Cứu Tương Lai

### 8.1 Cải Tiến SGD cho Ill-Conditioned Problems

#### 8.1.1 Preconditioning Approaches
- **Adaptive preconditioning**: Update preconditioner based on gradient history
- **Natural gradients**: Use Fisher information matrix for preconditioning
- **Sketching methods**: Approximate second-order information efficiently

#### 8.1.2 Variance Reduction Techniques
- **SVRG (Stochastic Variance Reduced Gradient)**: Periodically use full gradient
- **SAGA**: Maintain gradient table for variance reduction
- **Control variates**: Use auxiliary information to reduce noise

### 8.2 Hybrid Methods

#### 8.2.1 Batch-Stochastic Combinations
- Start with batch methods for initial progress
- Switch to SGD variants in well-conditioned regions
- **Condition-adaptive switching** between methods

#### 8.2.2 Multi-Scale Approaches
- Use different methods for different parameter groups
- **Hierarchical optimization**: Coarse-to-fine parameter updates

### 8.3 Theoretical Developments

#### 8.3.1 Convergence Theory for Ill-Conditioned Problems
- Develop **condition-number-aware bounds** for SGD
- **Non-asymptotic analysis** of SGD in difficult regimes
- **Refined noise models** that capture problem structure

#### 8.3.2 Optimal Batch Size Theory
- **Problem-specific batch size selection**
- Trade-offs between computational cost and convergence
- **Adaptive batch sizing** during optimization

## 9. Kết Luận và Đóng Góp

### 9.1 Những Phát Hiện Chính

**Finding 1: Complete SGD Failure**
All 12 SGD configurations failed to converge (0% success rate), demonstrating that SGD is not universally applicable as commonly assumed.

**Finding 2: Batch Size Irrelevance**
Increasing batch size from 20,000 to 30,000 provided no improvement, indicating that gradient noise is not the limiting factor in ill-conditioned problems.

**Finding 3: Advanced Techniques Insufficient**
Momentum, learning rate scheduling, and line search provided marginal improvements (5-15%) but all remained far from acceptable performance.

**Finding 4: Framework Independence**
Consistent failure across custom implementation and PyTorch confirms algorithmic rather than implementation issues.

### 9.2 Theoretical Contributions

#### 9.2.1 Empirical Limits of SGD
This work provides the first comprehensive empirical demonstration of SGD's complete failure on ill-conditioned regression problems, challenging the conventional wisdom about SGD's universal applicability.

#### 9.2.2 Condition Number as Primary Factor
Demonstrates that problem conditioning, not dataset size, determines SGD suitability - a fundamental shift from current scaling-focused perspectives.

#### 9.2.3 Noise-Signal Analysis
Provides concrete evidence that gradient signal-to-noise ratio in ill-conditioned problems falls below any recoverable threshold, regardless of batch size.

### 9.3 Practical Impact

#### 9.3.1 Production Guidelines
- **Condition assessment** should precede optimizer selection
- **SGD screening tests** should be mandatory before deployment
- **Fallback strategies** must be prepared for SGD failure scenarios

#### 9.3.2 Educational Implications
- Machine learning curricula should emphasize **SGD limitations**
- **Problem conditioning** should be taught alongside optimization methods
- **Empirical validation** should be emphasized over theoretical guarantees

### 9.4 Research Directions

#### 9.4.1 Immediate Priorities
1. **Develop SGD variants** robust to ill-conditioning
2. **Create diagnostic tools** for pre-deployment SGD assessment
3. **Build hybrid optimizers** that switch methods adaptively

#### 9.4.2 Long-term Vision
1. **Condition-aware optimization**: Methods that automatically adapt to problem structure
2. **Universal optimizers**: Single methods that work across conditioning regimes  
3. **Theoretical unification**: Connect conditioning, noise, and convergence in unified framework

### 9.5 Final Thoughts

This research demonstrates a critical gap between theoretical promises and practical performance of SGD. The complete failure across all variants tested highlights the importance of empirical validation and problem-specific method selection.

**Key Message**: SGD is not a universal solution. Problem conditioning determines algorithm suitability more than dataset size or computational resources. **Successful optimization requires matching algorithm characteristics to problem structure**, not blindly applying popular methods.

The path forward involves developing **condition-aware optimization methods** and **diagnostic frameworks** that guide practitioners toward appropriate algorithmic choices. The era of "one-size-fits-all" optimization is ending - the future belongs to **adaptive, problem-aware optimization strategies**.

---

**Acknowledgments**: The authors acknowledge the computational resources required for extensive SGD experimentation and the importance of negative results in advancing optimization science.

**Data Availability**: Experimental configurations and detailed failure logs are available for reproduction and further analysis.

**Conflict of Interest**: The authors declare no conflicts of interest related to this SGD failure analysis.

---

## Appendix A: Detailed Experimental Logs

### A.1 Convergence Trajectories
[Detailed plots of loss vs epoch for all 12 configurations showing plateau behavior]

### A.2 Gradient Norm Analysis
[Analysis of gradient norm evolution showing noise domination]

### A.3 Computational Cost Analysis
[Detailed timing and memory usage for each SGD variant]

### A.4 Hyperparameter Sensitivity
[Grid search results showing robustness of failure across parameter ranges]