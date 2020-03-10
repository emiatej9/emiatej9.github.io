---
layout: post
title: Effective Approaches to Attention-based Neural Machine Translation 
categories: [nlp]
---
이번 포스트는 [Effective Approaches to Attention-based Neural Machine Translation(MT Luong et al. 2015)](https://arxiv.org/pdf/1508.04025.pdf)
을 읽고 정리한 글입니다.

## Introduction
* NMT는 도메인 지식이 많이 필요하지 않다는 점, 개념적으로 단순하다는 점에서 매력적이다.
* Standard MT에 비해, 메모리 사용량이 적고 디코더 구현이 쉽다.
* 어텐션 기반 NMT는 Bahdanau et al.(2015) 이후 별 다른 연구가 이루어지지 않았다. 
* 이번 연구에서 효과적이고 간단한 어텐션 기반 NMT 모델을 `global`과 `local` 접근법으로 설계하였다.

## Neural Machine Translation
* NMT는 source $$x$$가 주어졌을 때 target $$y$$로 번역되는 확률 $$p(y \lvert x)$$을 모델링한 것이다.
* source에 대한 표현(representation) $$s$$를 계산하는 인코더, target을 출력하는 디코더로 구성된다.

$$
log \ p(y \lvert x)=\sum_{j=1}^m log \ p(y_j \lvert y_{<j}, s)
$$

* 최근 연구들은 RNN 아키텍처로 디코더를 모델링하면서, RNN 종류와 인코더 계산에서 방식을 달리한다.
* 디코더의 확률표현을 RNN 히든유닛 $$h_j$$과 vocabulary 차원의 벡터를 출력하는 함수 $$g$$로 나타낼 수 있다:

$$
p(y_j \lvert y_{<j}, s)=softmax(g(h_j)), \ \ \ h_j = f(h_{j-1}, s)
$$

* 함수 $$f$$는 이전`hidden state`로 현재`hidden state`를 구하는 RNN유닛, GRU, LSTM유닛이 될 수 있다.
* 이전 연구에서는 source에 대한 표현 $$s$$가 디코더의 `hidden state`초기화 시에만 사용이 되었으나, 
본 연구에서는 $$s$$가 전체 번역 과정에서 참조되는 어텐션 메커니즘을 활용하였다.
* (Sutskevere et al., 2014, Luong et al., 2015)를 따라, 이 연구에서도 LSTM 아키텍처를 쌓는 방식을 사용.
* 목적 함수는 다음과 같다:

$$
J_t = \sum_{(x,y) \in \mathbb{D}} -log \ p(y \lvert x)
$$

## Attention-based Models


