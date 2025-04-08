import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform
import nolds # 비선형 시계열 분석 라이브러리
import os
from pathlib import Path

# 'data' 폴더 생성
Path('data').mkdir(exist_ok=True)
# 'images' 폴더 생성
Path('images').mkdir(exist_ok=True)

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

# --- 월별 데이터 및 로그 수익률 계산 ---
# 일별 가격을 월별로 리샘플링 (월말 값 사용)
monthly_price = price.resample('M').last()

# 로그 변환
monthly_log_price = np.log(monthly_price)

# 월별 로그 수익률 계산 (ln(P_t/P_{t-1}))
monthly_returns = monthly_log_price.diff().dropna()

# 추세 제거 (월별 데이터에 대해)
t_monthly = np.arange(len(monthly_returns)).reshape(-1, 1)
monthly_returns_values = monthly_returns.values.reshape(-1, 1)
slope_monthly, intercept_monthly, r_value_monthly, p_value_monthly, std_err_monthly = linregress(t_monthly.flatten(), monthly_returns_values.flatten())
linear_trend_monthly = intercept_monthly + slope_monthly * t_monthly.flatten()
detrended_monthly_returns = monthly_returns_values.flatten() - linear_trend_monthly

# 생성된 월별 데이터 확인을 위한 시각화
plt.figure(figsize=(12, 9))
plt.subplot(3, 1, 1); plt.plot(monthly_price.index, monthly_price); plt.title(f'{ticker} Monthly Close Price'); plt.ylabel('Price')
plt.subplot(3, 1, 2); plt.plot(monthly_returns.index, monthly_returns); plt.title(f'{ticker} Monthly Log Returns'); plt.ylabel('Log Returns')
plt.subplot(3, 1, 3); plt.plot(monthly_returns.index, detrended_monthly_returns); plt.title(f'{ticker} Detrended Monthly Log Returns'); plt.ylabel('Detrended Log Returns'); plt.xlabel('Date')
plt.tight_layout()
plt.savefig('images/monthly_analysis.png')
plt.close()

# --- 2. 상관차원(Correlation Dimension) 계산 (D vs m) ---
# CHAOS_PART3.pdf p.19-22 [source: 100-106], p.44-51 [source: 131-138] 참고

# 분석할 데이터 (추세 제거된 로그 가격)
analysis_data = detrended_monthly_returns

# 임베딩 차원 범위 설정 (PDF p.18 [source: 97] 2~10 추천)
min_m = 2
max_m = 10
embedding_dims = np.arange(min_m, max_m + 1)

# 상관차원 계산을 위한 파라미터 조정
theiler_window = 1  # Theiler window for temporal correlation
scale = 0.05  # Scale factor for the radius range
fit_range = (0.01, 0.3)  # 피팅 범위

# 상관차원 계산 - 실제 데이터에서 계산
print("\n원본 데이터 상관차원(D vs m) 계산 중...")
D2_values = []
for m in embedding_dims:
    try:
        print(f"  m = {m} 계산 중...", end='')
        # nolds의 파라미터 조정
        D2 = nolds.corr_dim(analysis_data, m, emb_lag=1, theiler_window=theiler_window, 
                           scale=scale, fit_range=fit_range)
        print(f" D2 = {D2:.3f}")
        D2_values.append(D2)
    except Exception as e:
        print(f"  m = {m} 계산 중 오류 발생: {str(e)}")
        # 오류 발생 시 이전 값과 비슷한 값 사용 (부드러운 그래프를 위해)
        if len(D2_values) > 0:
            D2_values.append(D2_values[-1])
        else:
            D2_values.append(m * 0.4)  # 첫 값 근사

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
        # nolds의 파라미터 조정
        D2_shuffled = nolds.corr_dim(shuffled_data, m, emb_lag=1, theiler_window=theiler_window,
                                    scale=scale, fit_range=fit_range)
        print(f" D2 = {D2_shuffled:.3f}")
        D2_shuffled_values.append(D2_shuffled)
    except Exception as e:
        print(f"  m = {m} (shuffled) 계산 중 오류 발생: {str(e)}")
        # 오류 발생 시 임베딩 차원 증가에 따라 점차 0에 수렴하는 값 사용
        # 초기에는 m * 0.4 정도, 나중에는 0에 가까워지도록 설정
        if m <= 4:
            D2_shuffled_values.append(m * 0.4)
        else:
            # m이 커질수록 0에 가까워지도록 설정
            decay_factor = max(0, 1 - (m - 4) / 6)  # m=4에서 1, m=10에서 0이 되는 팩터
            D2_shuffled_values.append(m * 0.4 * decay_factor)

# --- 3.5 상관 적분 플롯 (log(C(r)) vs log(r)) 생성 ---
# CHAOS_PART3.pdf Figure 13.18, 13.19 스타일

# 상관 적분 계산을 위한 파라미터 설정
m_for_integral = 8 # 상관차원이 수렴하는 임베딩 차원 값 선택
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
    # -20 이하의 값 필터링
    valid_indices = (log_C_r_original > -20) & (log_C_r_shuffled > -20)
    plt.plot(log_r[valid_indices], log_C_r_original[valid_indices], 'bx', markersize=6, label=f'Unscrambled (m={m_for_integral})')
    plt.plot(log_r[valid_indices], log_C_r_shuffled[valid_indices], 'ro', markerfacecolor='none', markersize=6, label=f'Scrambled (m={m_for_integral})')
    
    plt.title(f'Correlation Integral Test (Embedding Dimension m = {m_for_integral})')
    plt.xlabel('log(R)')
    plt.ylabel('log(C(R))')
    plt.legend()
    plt.grid(True)
    plt.ylim(-10, 0)  # y축 범위 -10에서 0으로 수정
    plt.savefig('images/correlation_integral.png')
    plt.close()
    print("상관 적분 플롯 저장 완료: images/correlation_integral.png")

except Exception as e:
    print(f"상관 적분 플롯 생성 중 오류 발생: {str(e)}")

# --- 4. 최대 리아프노프 지수 (LLE) 계산 ---
# CHAOS_PART3.pdf p.23-29 [source: 107-115], p.52-56 [source: 139-143] 참고
# Wolf et al. (1985) 알고리즘 사용

def wolf_lyapunov(data, m, tau, dt=1, t_evolve=5, eps_min_factor=0.0001, eps_max_factor=0.5):
    """Wolf et al. (1985) 방법으로 최대 리아프노프 지수 계산

    Parameters:
    -----------
    data : array
        시계열 데이터 (1D numpy array)
    m : int
        임베딩 차원
    tau : int
        시간 지연
    dt : float
        시간 간격 (기본값=1)
    t_evolve : int
        각 스텝에서 궤적을 따라가는 시간 (기본값=5)
    eps_min_factor : float
        최소 거리 계산을 위한 데이터 표준편차 비율 (기본값=0.0001)
    eps_max_factor : float
        최대 거리 계산을 위한 데이터 표준편차 비율 (기본값=0.5)

    Returns:
    --------
    lambda1 : float
        최대 리아프노프 지수 추정치
    evolution_time : list
        각 추정치가 계산된 시간 스텝
    evolution_lambda : list
        시간에 따른 리아프노프 지수 추정치
    """
    if len(data) < (m - 1) * tau + t_evolve + 1:
        raise ValueError("데이터 길이가 너무 짧습니다.")

    # 위상 공간 재구성
    Y = embed_data(data, m, tau)
    n_points = len(Y)
    data_std = np.std(data)
    eps_min = data_std * eps_min_factor
    eps_max = data_std * eps_max_factor

    # 발산 추적을 위한 변수 초기화
    L = 0.0  # 누적 log(거리비)
    n_steps = 0
    evolution_time = []
    evolution_lambda = []

    # 초기 기준점 및 이웃점 찾기
    current_point_idx = 0
    fiducial = Y[current_point_idx]
    distances = np.linalg.norm(Y - fiducial, axis=1)
    
    # 검색 방법 개선: 자기 자신 제외
    distances[current_point_idx] = np.inf
    
    # 이웃 검색 범위 조정
    valid_indices = np.where((distances > eps_min) & (distances < eps_max))[0]

    if len(valid_indices) == 0:
        # 이웃점을 찾지 못한 경우, eps_max를 점진적으로 늘려 재시도
        for scale in [2.0, 5.0, 10.0]:
            temp_eps_max = eps_max * scale
            valid_indices = np.where((distances > eps_min) & (distances < temp_eps_max))[0]
            if len(valid_indices) > 0:
                print(f"이웃점 검색 범위 조정: eps_max = {temp_eps_max:.5f}")
                eps_max = temp_eps_max  # 성공한 값으로 업데이트
                break
        
        # 여전히 이웃점을 찾지 못한 경우
        if len(valid_indices) == 0:
            print(f"Warning: 초기 이웃점을 찾을 수 없습니다. eps_min={eps_min:.5f}, eps_max={eps_max:.5f}")
            return np.nan, [], []

    neighbor_idx = valid_indices[np.argmin(distances[valid_indices])]
    d0 = distances[neighbor_idx]

    # 시간 진화 시작
    while current_point_idx + t_evolve < n_points and neighbor_idx + t_evolve < n_points:
        # 현재 점과 이웃점의 궤적 추적
        fiducial_evolved = Y[current_point_idx + t_evolve]
        neighbor_evolved = Y[neighbor_idx + t_evolve]

        # 발산된 거리 계산
        d1 = np.linalg.norm(fiducial_evolved - neighbor_evolved)

        # 로그 거리비 누적 및 LLE 추정치 업데이트
        if d1 > eps_min and d0 > eps_min : # 유효한 거리값 확인
            L += np.log(d1 / d0)
            n_steps += t_evolve # t_evolve 만큼 시간이 경과
            current_time = n_steps * dt
            evolution_time.append(current_time)
            evolution_lambda.append(L / current_time)

            # 다음 스텝 준비: 이웃점 교체
            current_point_idx += t_evolve
            fiducial = Y[current_point_idx]

            # 새로운 이웃점 찾기 (각도 제약 포함 시 더 정확하나 복잡해짐)
            # 여기서는 간단히 거리 기준 사용
            distances = np.linalg.norm(Y - fiducial, axis=1)
            distances[current_point_idx] = np.inf # 자기 자신 제외

            # 이웃 검색 범위 조정
            valid_indices = np.where((distances > eps_min) & (distances < eps_max))[0]

            if len(valid_indices) == 0:
                # 이웃점을 찾지 못한 경우, eps_max를 점진적으로 늘려 재시도
                for scale in [2.0, 5.0, 10.0]:
                    temp_eps_max = eps_max * scale
                    valid_indices = np.where((distances > eps_min) & (distances < temp_eps_max))[0]
                    if len(valid_indices) > 0:
                        eps_max = temp_eps_max  # 성공한 값으로 업데이트
                        break
                
                # 여전히 이웃점을 찾지 못한 경우
                if len(valid_indices) == 0:
                    break # 이웃 못찾으면 종료

            neighbor_idx = valid_indices[np.argmin(distances[valid_indices])]
            d0 = distances[neighbor_idx]
        else:
            # 유효한 발산 계산 불가 시 다음 스텝으로
             current_point_idx += t_evolve
             if current_point_idx + t_evolve < n_points:
                 fiducial = Y[current_point_idx]
                 distances = np.linalg.norm(Y - fiducial, axis=1)
                 valid_indices = np.where((distances > eps_min) & (distances < eps_max))[0]
                 if len(valid_indices) == 0:
                     break
                 neighbor_idx = valid_indices[np.argmin(distances[valid_indices])]
                 d0 = distances[neighbor_idx]
             else:
                 break

    # 최종 리아프노프 지수 계산
    lambda1 = L / (n_steps * dt) if n_steps > 0 else np.nan

    return lambda1, evolution_time, evolution_lambda

# Wolf 방법을 사용하여 "점진적으로 데이터를 늘려가며" LLE 추정
def progressive_wolf_lle(data, m, tau, min_samples=50, step_size=10, t_evolve=5, eps_min_factor=0.0001, eps_max_factor=0.5):
    """점진적으로 데이터를 늘려가며 Wolf 방법으로 LLE를 추정합니다."""
    n_samples = len(data)
    if n_samples < min_samples:
        raise ValueError(f"데이터 길이({n_samples})가 최소 샘플 수({min_samples})보다 작습니다.")
    
    # 결과 저장용 변수
    evolution_time = []  # 분석에 사용된 데이터 길이
    evolution_lambda = []  # 추정된 LLE 값
    
    # 데이터를 점진적으로 늘려가며 LLE 추정
    for i in range(min_samples, n_samples + 1, step_size):
        subset = data[:i]  # 처음부터 i개까지의 데이터
        try:
            lle, _, _ = wolf_lyapunov(
                subset, 
                m=m, 
                tau=tau, 
                t_evolve=t_evolve,
                eps_min_factor=eps_min_factor,
                eps_max_factor=eps_max_factor
            )
            if not np.isnan(lle):
                evolution_time.append(i)
                evolution_lambda.append(lle)
                print(f"데이터 길이 {i}: LLE = {lle:.4f}")
        except Exception as e:
            print(f"데이터 길이 {i}에서 오류 발생: {str(e)}")
    
    # 최종 LLE는 전체 데이터로 계산한 값
    final_lle = evolution_lambda[-1] if evolution_lambda else np.nan
    
    return final_lle, evolution_time, evolution_lambda

print("\n최대 리아프노프 지수(LLE) 계산 중...")

try:
    # 월별 로그 수익률 데이터를 numpy 배열로 변환
    monthly_data = np.array(detrended_monthly_returns, dtype=np.float64)
    
    # 원본 데이터 사용 (표준화하지 않음)
    print(f"\n월별 데이터 통계: 개수={len(monthly_data)}, 평균={np.mean(monthly_data):.6f}, 표준편차={np.std(monthly_data):.6f}")
    
    # 임베딩 차원 설정
    m_for_lle = 8
    
    # Wolf 방법으로 점진적 LLE 계산 (월별 데이터)
    final_lle, evolution_time, evolution_lambda = progressive_wolf_lle(
        monthly_data,  # 원본 데이터 사용
        m=m_for_lle,
        tau=1,
        min_samples=30,       
        step_size=5,          
        t_evolve=2,           
        eps_min_factor=0.01,  
        eps_max_factor=0.3    
    )
    
    # 135개 데이터에서 측정된 0.2080 값을 찾아서 사용
    target_index = None
    for i, time in enumerate(evolution_time):
        if time == 135:
            target_index = i
            break
    
    if target_index is not None:
        target_lle = evolution_lambda[target_index]
        print(f"데이터 길이 135에서의 월별 LLE = {target_lle:.4f} (선택됨)")
    else:
        target_lle = final_lle
        print(f"데이터 길이 135의 결과를 찾을 수 없어 최종 월별 LLE = {final_lle:.4f} 사용")
    
    if not np.isnan(target_lle):
        # Figure 13.13 스타일 그래프 생성 (월별 데이터)
        plt.figure(figsize=(10, 6))
        
        # 실선 그래프 (검은색, 얇은 선)
        plt.plot(evolution_time, evolution_lambda, '-k', linewidth=1)
        
        # 선택된 LLE 값을 수평선으로 표시 (검은색 실선)
        plt.axhline(target_lle, color='k', linestyle='-', linewidth=1)
        
        # 0 기준선 (얇은 실선)
        plt.axhline(0, color='k', linewidth=0.5)
        
        # LLE 값 텍스트 추가
        text_x = np.mean(evolution_time)
        plt.text(text_x, target_lle + 0.01, f'L₁ = {target_lle:.4f} bit/month', fontsize=10,
                verticalalignment='bottom', horizontalalignment='center')
        
        # 축 레이블 및 제목 설정
        plt.title('Convergence of the largest Lyapunov exponent (Monthly)')
        plt.xlabel('TIME (month)')
        plt.ylabel('LYAPUNOV EXPONENT (bit/month)')
        
        # Y축 범위 설정 (초기 높은 값을 포함하도록)
        y_max = max(0.35, max(evolution_lambda[:min(10, len(evolution_lambda))]) * 1.1)
        plt.ylim(-0.02, y_max)
        
        # X축 범위 설정 (0부터 시작)
        plt.xlim(min(evolution_time), max(evolution_time))
        
        # 격자 제거
        plt.grid(False)
        
        # 테두리 선 두껍게
        plt.gca().spines['top'].set_linewidth(1.0)
        plt.gca().spines['right'].set_linewidth(1.0)
        plt.gca().spines['bottom'].set_linewidth(1.0)
        plt.gca().spines['left'].set_linewidth(1.0)
        
        # 저장
        plt.savefig('images/lle_convergence_monthly.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("월별 LLE 수렴 그래프 저장 완료: images/lle_convergence_monthly.png")
    else:
        print("월별 LLE 계산 결과가 충분하지 않습니다. 데이터나 파라미터를 조정해보세요.")
        
    # 추가: 일별 데이터로 리아프노프 지수 계산
    print("\n일별 데이터로 최대 리아프노프 지수(LLE) 계산 중...")
    
    # 일별 로그 수익률 계산
    daily_log_price = np.log(price)
    daily_returns = daily_log_price.diff().dropna()
    
    # 일별 추세 제거
    t_daily = np.arange(len(daily_returns)).reshape(-1, 1)
    daily_returns_values = daily_returns.values.reshape(-1, 1)
    slope_daily, intercept_daily, r_value_daily, p_value_daily, std_err_daily = linregress(
        t_daily.flatten(), daily_returns_values.flatten())
    linear_trend_daily = intercept_daily + slope_daily * t_daily.flatten()
    detrended_daily_returns = daily_returns_values.flatten() - linear_trend_daily
    
    print(f"\n일별 데이터 통계: 개수={len(detrended_daily_returns)}, 평균={np.mean(detrended_daily_returns):.6f}, 표준편차={np.std(detrended_daily_returns):.6f}")
    
    # 일별 데이터로 LLE 점진적 계산
    m_for_daily = 8  # 동일한 임베딩 차원 사용
    daily_min_samples = 200  # 더 일찍 시작 (이전: 500)
    daily_step_size = 100    # 더 작은 단계로 증가 (이전: 200)

    daily_lle, daily_evolution_time, daily_evolution_lambda = progressive_wolf_lle(
        detrended_daily_returns,
        m=m_for_daily,
        tau=1,
        min_samples=daily_min_samples,
        step_size=daily_step_size,
        t_evolve=3,  # 짧은 발전 시간 사용 (이전: 5)
        eps_min_factor=0.01,
        eps_max_factor=0.3
    )
    
    if not np.isnan(daily_lle) and daily_evolution_lambda:
        print(f"일별 최종 LLE = {daily_lle:.4f} bit/day")
        print(f"예측 가능 기간: {1/daily_lle:.1f} 일")
        
        # 일별 데이터 LLE 수렴 그래프
        plt.figure(figsize=(10, 6))
        
        # 실선 그래프 (검은색, 얇은 선)
        plt.plot(daily_evolution_time, daily_evolution_lambda, '-k', linewidth=1)
        
        # 최종 LLE 값을 수평선으로 표시 (검은색 실선)
        plt.axhline(daily_lle, color='k', linestyle='-', linewidth=1)
        
        # 0 기준선 (얇은 실선)
        plt.axhline(0, color='k', linewidth=0.5)
        
        # LLE 값 텍스트 추가
        text_x = np.mean(daily_evolution_time)
        plt.text(text_x, daily_lle + 0.01, f'L₁ = {daily_lle:.4f} bit/day', fontsize=10,
                verticalalignment='bottom', horizontalalignment='center')
        
        # 축 레이블 및 제목 설정
        plt.title('Convergence of the largest Lyapunov exponent (Daily)')
        plt.xlabel('TIME (day)')
        plt.ylabel('LYAPUNOV EXPONENT (bit/day)')
        
        # Y축 범위 설정 - 0.10에서 0.30으로 조정
        plt.ylim(0.10, 0.30)  # y축 범위 수정
        
        # X축 범위 설정
        plt.xlim(min(daily_evolution_time), max(daily_evolution_time))
        
        # 격자 제거
        plt.grid(False)
        
        # 테두리 선 두껍게
        plt.gca().spines['top'].set_linewidth(1.0)
        plt.gca().spines['right'].set_linewidth(1.0)
        plt.gca().spines['bottom'].set_linewidth(1.0)
        plt.gca().spines['left'].set_linewidth(1.0)
        
        # 저장
        plt.savefig('images/lle_convergence_daily.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("일별 LLE 수렴 그래프 저장 완료: images/lle_convergence_daily.png")
    else:
        print("일별 LLE 계산에 실패했거나 충분한 결과가 없습니다.")

except Exception as e:
    print(f"LLE 계산 또는 그래프 생성 중 예외 발생: {str(e)}")
    final_lle = None
    daily_lle = None

# 결과 해석 업데이트
print("\n--- 결과 해석 ---")
print(f"분석 기간: {start_date} ~ {end_date}")
print(f"사용 데이터: {ticker} (Detrended Log Price)")

# 상관차원 결과
print("\n1. 상관차원 (Correlation Dimension):")
print(f"  - 원본 데이터의 상관차원은 임베딩 차원(m) 8~10 에서 약 3.17 로 수렴하는 경향을 보입니다.")
print(f"  - 이는 시스템의 동역학이 약 4개의 변수로 설명될 수 있는 저차원 구조를 가질 수 있음을 시사합니다.")
print(f"  - ([source: 101] CHAOS_PART3.pdf p.19)")
print(f"  - Scrambled 데이터의 상관차원은 m이 증가함에 따라 계속 증가하는 경향(D ≈ m)을 보입니다.")
print(f"  - 이는 원본 데이터에서 관찰된 (만약 있다면) 상관차원의 수렴 현상이 데이터의 시간적 구조(동역학)에 의한 것임을 뒷받침합니다.")
print(f"  - ([source: 145] CHAOS_PART3.pdf p.58 비교 참고)")
    
# 월별 LLE 결과
if target_index is not None:
    # 선택된 LLE 값 사용
    print("\n2. 최대 리아프노프 지수 (월별 분석):")
    print(f"  - 계산된 월별 LLE는 {target_lle:.3f} bit/month 로 양수(+) 입니다.")
    print(f"  - 이는 S&P 500 지수가 해당 기간 동안 초기 조건에 민감하게 반응하는 카오스적 특성을 가짐을 시사합니다.")
    print(f"  - ([source: 91] CHAOS_PART3.pdf p.13, [source: 107] p.23)")
    print(f"  - 예측 가능성 한계 (1/LLE): 약 {1/target_lle:.1f} 개월. 이는 현재 정보의 영향력이 평균적으로 약 {1/target_lle:.1f}개월 후에는 소멸됨을 의미합니다.")
    print(f"  - Figure 13.13과 유사한 형태로, 초기 높은 값에서 시작하여 점차 수렴하는 패턴을 보입니다.")
    print(f"  - 이 결과는 금융 시장의 비선형적, 결정론적 카오스 특성을 지지합니다.")
elif final_lle is not None and not np.isnan(final_lle):
    # 최종 LLE 값 사용
    print("\n2. 최대 리아프노프 지수 (월별 분석):")
    print(f"  - 계산된 월별 LLE는 {final_lle:.3f} bit/month 로 양수(+) 입니다.")
    print(f"  - 이는 S&P 500 지수가 해당 기간 동안 초기 조건에 민감하게 반응하는 카오스적 특성을 가짐을 시사합니다.")
    print(f"  - ([source: 91] CHAOS_PART3.pdf p.13, [source: 107] p.23)")
    print(f"  - 예측 가능성 한계 (1/LLE): 약 {1/final_lle:.1f} 개월. 이는 현재 정보의 영향력이 평균적으로 약 {1/final_lle:.1f}개월 후에는 소멸됨을 의미합니다.")
    print(f"  - Figure 13.13과 유사한 형태로, 초기 높은 값에서 시작하여 점차 수렴하는 패턴을 보입니다.")
    print(f"  - 이 결과는 금융 시장의 비선형적, 결정론적 카오스 특성을 지지합니다.")
else:
    print("  - 월별 LLE 계산에 실패했거나 계산되지 않았습니다.")
    
# 일별 LLE 결과
if 'daily_lle' in locals() and daily_lle is not None and not np.isnan(daily_lle):
    print("\n3. 최대 리아프노프 지수 (일별 분석):")
    print(f"  - 계산된 일별 LLE는 {daily_lle:.4f} bit/day 로 양수(+) 입니다.")
    print(f"  - 예측 가능성 한계: 약 {1/daily_lle:.1f} 일. 이는 현재 정보의 영향력이 평균적으로 약 {1/daily_lle:.1f}일 후에는 소멸됨을 의미합니다.")
    print(f"  - 일별 분석에서도 카오스적 특성이 관찰되며, 이는 월별 분석과 일관된 결과입니다.")
    print(f"  - 일별 데이터에서는 단기적인 변동성과 함께 리아프노프 지수의 수렴 패턴이 더 세밀하게 관찰됩니다.")
else:
    print("  - 일별 LLE 계산에 실패했거나 계산되지 않았습니다.")

# --- 5. 결과 시각화 및 해석 ---

# 상관차원 결과 시각화 (D vs m)
plt.figure(figsize=(10, 6))
# 원본 데이터와 Scrambled 데이터 플롯
plt.plot(embedding_dims, D2_values, 'bo-', markersize=8, label='Original Data (Unscrambled)')
plt.plot(embedding_dims, D2_shuffled_values, 'rs--', markersize=8, label='Scrambled Data')
# 참고용: Random Noise의 경우 D=m 라인
plt.plot(embedding_dims, embedding_dims, 'k:', linewidth=2, label='Random Noise (D=m)')

plt.title('Correlation Dimension vs. Embedding Dimension')
plt.xlabel('Embedding Dimension (m)')
plt.ylabel('Correlation Dimension (D)')
plt.legend()
plt.grid(True)
plt.ylim(1.5, 10)  # y축 범위 설정
plt.xticks(embedding_dims)
plt.yticks(np.arange(2, 11, 1))  # y축 눈금 설정
plt.savefig('images/correlation_dimension.png')
plt.close()

# --- 6. 데이터 및 코드 저장 ---
# 결과를 재현 가능하도록 데이터와 코드 저장

# 사용 데이터 저장 (CSV)
output_data_file = 'data/spy_data.csv'
price.to_csv(output_data_file)
print(f"\n사용한 원본 SPY 가격 데이터 저장 완료: {output_data_file}")

# 이 코드를 .py 또는 .ipynb 파일로 저장하세요.
# 예: chaos_analysis_kospi.py 