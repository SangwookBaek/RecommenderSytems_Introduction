#!/bin/bash

# wget 설치 여부 확인
if ! command -v wget &> /dev/null
then
    echo "Error: wget이 설치되어 있지 않습니다. 먼저 wget을 설치해주세요."
    exit 1
fi

# 데이터 디렉토리 만들기
mkdir -p ./data

# 파일 다운로드
wget -nc --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-10m.zip -P ./data

# 압축 해제
unzip -n ./data/ml-10m.zip -d ./data/