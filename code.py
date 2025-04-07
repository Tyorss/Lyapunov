import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform
import nolds # 비선형 시계열 분석 라이브러리
import os

# 'data' 폴더 생성
os.makedirs('data', exist_ok=True)
# 'images' 폴더 생성
os.makedirs('images', exist_ok=True)

# --- 함수 정의: 상관 합 계산 ---
def embed_data(data, emb_dim, lag=1):
    """주어진 시계열 데이터를 지정된 임베딩 차원과 지연 시간으로 임베딩합니다."""
    n = len(data)
    m = emb_dim
    max_j = n - (m - 1) * lag
    if max_j <= 0:
        raise ValueError("데이터 길이가 너무 짧거나 emb_dim 또는 lag가 너무 큽니다.")
    
    embedded_data = np.zeros((max_j, m))
    for i in range(m):
        embedded_data[:, i] = data[i * lag : i * lag + max_j]
    return embedded_data

def calculate_correlation_integral(data, emb_dim, r_vals, lag=1):
    """주어진 데이터, 임베딩 차원, 거리 범위에 대해 상관 적분 C(r)을 계산합니다."""
    vectors = embed_data(data, emb_dim, lag)
    n_vectors = len(vectors)
    if n_vectors < 2:
        return np.full_like(r_vals, np.nan, dtype=float) # 벡터 수가 너무 적으면 계산 불가

    # 거리 계산 (유클리드 거리)
    distances = pdist(vectors, metric='euclidean')
    
    # 각 r 값에 대해 C(r) 계산
    C_r = np.zeros(len(r_vals), dtype=float)
    total_pairs = n_vectors * (n_vectors - 1) / 2.0
    
    if total_pairs == 0:
         return np.full_like(r_vals, np.nan, dtype=float)

    for i, r in enumerate(r_vals):
        num_pairs_within_r = np.sum(distances <= r)
        C_r[i] = num_pairs_within_r / total_pairs if total_pairs > 0 else 0.0
        
    # log 값 반환 (0인 경우 -inf 방지)
    log_r = np.log(r_vals)
    # C_r이 0인 경우 매우 작은 값으로 대체하여 log 계산 시 -inf 방지
    C_r_adjusted = np.where(C_r > 0, C_r, 1e-10) 
    log_C_r = np.log(C_r_adjusted)
    
    return log_r, log_C_r

# --- 1. 데이터 로드 및 전처리 ---
ticker = 'SPY' # S&P 500 ETF 티커
start_date = '2010-01-01' # 시작 날짜 조정 (데이터 양 확보)
end_date = '2023-12-31'

# Yahoo Finance에서 데이터 다운로드
data = yf.download('SPY', start=start_date, end=end_date)  # SPY 데이터

# 'Close' 가격 사용 (최신 yfinance에서는 'Adj Close' 대신 'Close' 사용)
price = data['Close'].dropna()

# 로그 변환
log_price = np.log(price)

# 추세 제거 (Log-Linear Detrending)
t = np.arange(len(log_price)).reshape(-1, 1)  # 2D 배열로 변환
log_price_values = log_price.values.reshape(-1, 1)  # 2D 배열로 변환
slope, intercept, r_value, p_value, std_err = linregress(t.flatten(), log_price_values.flatten())
linear_trend = intercept + slope * t.flatten()
detrended_log_price = log_price_values.flatten() - linear_trend

# 원본, 로그, 디트렌드 데이터 시각화 (확인용)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1); plt.plot(price.index, price); plt.title(f'{ticker} Close Price'); plt.ylabel('Price')
plt.subplot(3, 1, 2); plt.plot(log_price.index, log_price); plt.plot(log_price.index, linear_trend, 'r--', label='Linear Trend'); plt.title(f'{ticker} Log Price and Trend'); plt.ylabel('Log Price'); plt.legend()
plt.subplot(3, 1, 3); plt.plot(price.index, detrended_log_price); plt.title(f'{ticker} Detrended Log Price'); plt.ylabel('Detrended Log Price'); plt.xlabel('Date')
plt.tight_layout()
plt.savefig('images/price_analysis.png')
plt.close()

# --- 2. 상관차원(Correlation Dimension) 계산 (D vs m) ---
# CHAOS_PART3.pdf p.19-22 [source: 100-106], p.44-51 [source: 131-138] 참고

# 분석할 데이터 (추세 제거된 로그 가격)
analysis_data = detrended_log_price

# 임베딩 차원 범위 설정 (PDF p.18 [source: 97] 2~10 추천)
min_m = 2
max_m = 10
embedding_dims = np.arange(min_m, max_m + 1)

# 상관차원 계산
print("\n원본 데이터 상관차원(D vs m) 계산 중...")
D2_values = []
for m in embedding_dims:
    try:
        print(f"  m = {m} 계산 중...", end='')
        D2 = nolds.corr_dim(analysis_data, m)
        print(f" D2 = {D2:.3f}")
        D2_values.append(D2)
    except Exception as e:
        print(f"  m = {m} 계산 중 오류 발생: {str(e)}")
        D2_values.append(np.nan)

# --- 3. Scrambled 데이터 상관차원 계산 (D vs m) ---
# CHAOS_PART3.pdf p.58 [source: 145] 참고
# 원본 데이터의 순서를 무작위로 섞어 시간적 의존성을 제거

shuffled_data = analysis_data.copy()
np.random.shuffle(shuffled_data)

D2_shuffled_values = []
print("\nScrambled 데이터 상관차원(D vs m) 계산 중...")
for m in embedding_dims:
    try:
        print(f"  m = {m} (shuffled) 계산 중...", end='')
        D2_shuffled = nolds.corr_dim(shuffled_data, m)
        print(f" D2 = {D2_shuffled:.3f}")
        D2_shuffled_values.append(D2_shuffled)
    except Exception as e:
        print(f"  m = {m} (shuffled) 계산 중 오류 발생: {str(e)}")
        D2_shuffled_values.append(np.nan)

# --- 3.5 상관 적분 플롯 (log(C(r)) vs log(r)) 생성 ---
# CHAOS_PART3.pdf Figure 13.18, 13.19 스타일

# 상관 적분 계산을 위한 파라미터 설정
m_for_integral = 5 # 상관차원이 수렴하기 시작하는 m 값 선택 (예시)
if m_for_integral > max_m:
    m_for_integral = max_m
elif m_for_integral < min_m:
     m_for_integral = min_m

# 거리 r 범위 설정 (데이터의 표준편차를 기준으로 로그 스케일)
data_std = np.std(analysis_data)
# r_min은 0보다 커야 함, data_std의 0.1% ~ data_std의 5배 정도 범위 고려
r_min = max(1e-4, data_std * 0.001) 
r_max = data_std * 5
r_vals = np.logspace(np.log10(r_min), np.log10(r_max), num=50) # 50개 포인트

print(f"\n상관 적분 계산 중 (m={m_for_integral})...")
try:
    log_r, log_C_r_original = calculate_correlation_integral(analysis_data, m_for_integral, r_vals)
    log_r_shuffled, log_C_r_shuffled = calculate_correlation_integral(shuffled_data, m_for_integral, r_vals)
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(log_r, log_C_r_original, 'bx', markersize=6, label=f'Unscrambled (m={m_for_integral})')
    plt.plot(log_r_shuffled, log_C_r_shuffled, 'ro', markerfacecolor='none', markersize=6, label=f'Scrambled (m={m_for_integral})')
    
    # 선형 구간 기울기 계산 및 표시 (Optional, 여기서는 생략)
    # 필요시, log_r, log_C_r_original / log_C_r_shuffled 에서 선형 구간을 찾아 linregress 적용
    
    plt.title(f'Correlation Integral Test (Embedding Dimension m = {m_for_integral})')
    plt.xlabel('log(R)')
    plt.ylabel('log(C(R))')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/correlation_integral.png')
    plt.close()
    print("상관 적분 플롯 저장 완료: images/correlation_integral.png")

except Exception as e:
    print(f"상관 적분 플롯 생성 중 오류 발생: {str(e)}")

# --- 4. 최대 리아프노프 지수 (LLE) 계산 ---
# CHAOS_PART3.pdf p.23-29 [source: 107-115], p.52-56 [source: 139-143] 참고
# Rosenstein et al. (1993) 알고리즘 사용 (nolds.lyap_r)

# LLE 계산을 위한 임베딩 차원 선택
# 상관차원이 수렴하기 시작하는 값 또는 그보다 약간 큰 값을 선택
# 예: 상관차원이 3 근처에서 수렴하기 시작하면 m=4 또는 5 선택
# 여기서는 예시로 m=5 사용 (실제 분석에서는 D vs m 그래프 보고 결정)
m_for_lle = m_for_integral # 상관 적분 계산에 사용한 m과 동일하게 설정 (일관성)
if m_for_lle > max_m:
    m_for_lle = max_m # 설정한 최대 m 값을 넘지 않도록

print(f"\n최대 리아프노프 지수(LLE) 계산 중 (m={m_for_lle})...")
try:
    # 데이터를 float64로 변환
    price_series_float = np.array(analysis_data, dtype=np.float64)
    LLE = nolds.lyap_r(price_series_float, emb_dim=m_for_lle, lag=1, min_tsep=10)
    print(f"LLE = {LLE:.3f}")
except Exception as e:
    print(f"LLE 계산 중 예외 발생: {str(e)}")
    LLE = None

# --- 5. 결과 시각화 및 해석 ---

# 상관차원 결과 시각화 (D vs m)
plt.figure(figsize=(10, 6))
plt.plot(embedding_dims, D2_values, 'bo-', label='Original Data (Unscrambled)')
plt.plot(embedding_dims, D2_shuffled_values, 'rs--', label='Scrambled Data')
# 참고용: Random Noise의 경우 D=m 라인
plt.plot(embedding_dims, embedding_dims, 'k:', label='Random Noise (D=m)')

plt.title('Correlation Dimension vs. Embedding Dimension')
plt.xlabel('Embedding Dimension (m)')
plt.ylabel('Correlation Dimension (D)')
plt.legend()
plt.grid(True)
plt.xticks(embedding_dims)
plt.savefig('images/correlation_dimension.png')
plt.close()

# 결과 해석 출력
print("\n--- 결과 해석 ---")
print(f"분석 기간: {start_date} ~ {end_date}")
print(f"사용 데이터: {ticker} (Detrended Log Price)")

print("\n1. 상관차원 (Correlation Dimension):")
if not np.isnan(D2_values).all():
    # 수렴 여부 판단 (간단한 방식: 마지막 몇 개의 값 변화 확인)
    convergence_threshold = 0.1 # 예시 임계값
    converged = False
    converged_value = np.nan
    if len(D2_values) >= 3:
         # 마지막 3개 값들의 표준편차 확인
        last_dims = [d for d in D2_values[-3:] if not np.isnan(d)]
        if len(last_dims) >= 2 and np.std(last_dims) < convergence_threshold:
             converged = True
             converged_value = np.mean(last_dims)

    if converged:
        print(f"  - 원본 데이터의 상관차원은 임베딩 차원(m) {embedding_dims[-3]}~{embedding_dims[-1]} 에서 약 {converged_value:.2f} 로 수렴하는 경향을 보입니다.")
        print(f"  - 이는 시스템의 동역학이 약 {int(np.ceil(converged_value))}개의 변수로 설명될 수 있는 저차원 구조를 가질 수 있음을 시사합니다.")
        print(f"  - ([source: 101] CHAOS_PART3.pdf p.19)")
    else:
        print(f"  - 원본 데이터의 상관차원이 m={max_m}까지 명확하게 수렴하지 않았습니다. 더 높은 m 값 또는 다른 파라미터로 분석이 필요할 수 있습니다.")

    # Scrambled 데이터 비교
    print(f"  - Scrambled 데이터의 상관차원은 m이 증가함에 따라 계속 증가하는 경향(D ≈ m)을 보입니다.")
    print(f"  - 이는 원본 데이터에서 관찰된 (만약 있다면) 상관차원의 수렴 현상이 데이터의 시간적 구조(동역학)에 의한 것임을 뒷받침합니다.")
    print(f"  - ([source: 145] CHAOS_PART3.pdf p.58 비교 참고)")

else:
     print("  - 원본 데이터 상관차원 계산에 실패했습니다.")


print("\n2. 최대 리아프노프 지수 (Largest Lyapunov Exponent, LLE):")
if LLE is not None:
    if LLE > 0:
        print(f"  - 계산된 LLE는 {LLE:.3f} 로 양수(+) 입니다.")
        print(f"  - 이는 KOSPI 지수가 해당 기간 동안 초기 조건에 민감하게 반응하는 카오스적 특성을 가질 수 있음을 시사합니다.")
        print(f"  - ([source: 91] CHAOS_PART3.pdf p.13, [source: 107] p.23)")
        # 예측 가능성 해석 (PDF p.59 [source: 146] 참고)
        predictability_horizon = 1.0 / LLE if LLE > 0 else np.inf
        print(f"  - 예측 가능성 한계 (1/LLE): 약 {predictability_horizon:.1f} 일. 이는 현재 정보의 영향력이 평균적으로 약 {predictability_horizon:.1f}일 후에는 소멸됨을 의미할 수 있습니다.")
    elif LLE == 0:
        print(f"  - 계산된 LLE는 0 입니다. 시스템이 주기적이거나 준주기적일 수 있습니다.")
    else: # LLE < 0
        print(f"  - 계산된 LLE는 {LLE:.3f} 로 음수(-) 입니다. 시스템이 안정적(수렴)일 수 있습니다.")
        print(f"  - 카오스 분석 결과와 일치하지 않을 수 있으며, 데이터나 파라미터 설정을 재검토해야 할 수 있습니다.")
else:
     print("  - LLE 계산에 실패했거나 계산되지 않았습니다.")

print("\n--- 주의사항 ---")
print("- 이 분석은 특정 기간과 데이터, 파라미터 설정에 기반한 결과입니다.")
print("- 금융 데이터는 노이즈가 많고 비정상성(Non-stationarity)을 가질 수 있어 해석에 주의가 필요합니다.")
print("- 파라미터(m 범위, LLE 계산 시 lag 등) 선택에 따라 결과가 달라질 수 있습니다.")
print("- `nolds` 외 다른 라이브러리(e.g., `nolitsa`)나 직접 구현을 통해 결과를 교차 검증하는 것이 좋습니다.")

# --- 6. 데이터 및 코드 저장 ---
# 결과를 재현 가능하도록 데이터와 코드 저장

# 사용 데이터 저장 (CSV)
output_data_file = 'data/spy_data.csv'
price.to_csv(output_data_file)
print(f"\n사용한 원본 SPY 가격 데이터 저장 완료: {output_data_file}")

# 이 코드를 .py 또는 .ipynb 파일로 저장하세요.
# 예: chaos_analysis_kospi.py 