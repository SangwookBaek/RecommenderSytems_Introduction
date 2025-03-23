> 🇺🇸 [View in English](./README_eng.md)


# 추천 시스템 (최신화 및 최적화 버전)

이 저장소는 [moseskim/RecommenderSystems](https://github.com/moseskim/RecommenderSystems)를 기반으로 최신 python 환경에 맞게 업데이트하고 최적화한 프로젝트입니다. 교육용 코드를 유지하면서, macOS M1, Linux AARCH 등 다양한 시스템에서 원활히 작동하도록 개선하였습니다.

## 주요 업데이트

- **Python 3.10** 기반으로 업데이트
- 구형 및 미지원 패키지 제거
- 다양한 시스템 아키텍처 (x86, ARM 등)에서 호환성 향상
- 다양한 데이터 로딩 방식 추가:
  - **Pandas** (기본 버전)
  - **Dask** (병렬 및 대용량 처리)
  - **PyArrow** (고속 I/O 및 메모리 최적화)

## 개발 환경

- **Python Version**: 3.10
- **Recommended Environment**: `venv` or `conda`
- Works across macOS (Intel & M-series), Linux (x86 & ARM)

## 필요 패키지

- `pandas`
- `numpy`
- `matplotlib`
- `dask`
- `pyarrow`

```bash
pip install pandas numpy matplotlib dask pyarrow
```

> ⚠️ **주의:** 특정 버전에 얽매일 필요는 없습니다. 일반적인 범용 패키지를 사용하기 때문에, 본인의 Python 환경에 맞게 설치해도 무방합니다.  
참고로 작성자는 **Apple M1 Pro**, **Python 3.10 + Conda** 환경에서 프로젝트를 개발 및 테스트했습니다.



## 최적화 내용

### ✅ 데이터 로딩
- [x] Dask로 대용량 CSV 처리 개선
- [x] PyArrow 기반 고속 I/O 구현
- 벤치마크 성능 비교 (coming soon)

### 🛠️ 모델 성능 향상 예정
(coming soon)

### 📦 Packaging & CI
(coming soon)

## References

- Original Repo: [moseskim/RecommenderSystems](https://github.com/moseskim/RecommenderSystems)



