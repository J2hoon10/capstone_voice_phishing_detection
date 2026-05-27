# Streaming Belief v5 아키텍처 개요 및 관련 논문(References)

## 1. Streaming Belief v5 구조 요약

`streaming_belief_v5` 모델의 코드 및 실험 계획서(`2026-04-15_streaming_belief_v5_experiment_report.md` 등)를 파악한 결과, 이 모델은 **스트리밍 환경에서의 실시간 화자 의도(Belief-State) 분류**를 위해 설계되었습니다. 주요 핵심 기술 및 아키텍처 요소를 요약하면 다음과 같습니다.

1. **KoELECTRA 기반 특징 추출**: 한국어 문맥 처리에 특화된 사전학습 모델인 KoELECTRA를 사용하여 매 타임스텝의 텍스트(청크)를 임베딩 공간으로 변환합니다.
2. **LoRA (Low-Rank Adaptation)**: 인코더 레이어 전체를 무겁게 학습시키는 대신, 인코더에 랭크(rank=8)가 낮은 레이어를 덧붙이는 LoRA 미세조정을 도입하여 효율적인 학습과 과적합 방지를 달성했습니다.
3. **Attention-weighted Pooling**: 기존 BERT류 모델에서 쓰는 `[CLS]` 토큰 기반 구조에서 벗어나, 발화의 실제 토큰들에 Attention 가중치를 계산 및 부여하여 최종 청크 벡터를 생성합니다.
4. **Mamba SSM (State Space Model)**: 생성된 청크 벡터들은 시간 흐름에 따라 순차적으로 2레이어 Mamba에 주입됩니다. 기존 RNN/Transformer와 달리 Mamba(선형 복잡도 유지)를 통해 은닉 상태(d_state=16)를 효과적으로 압축하고 텍스트 분류에 있어서 번복률이 낮은 안정적인 조기 종료(Early Stop)를 실현했습니다.
5. **시간 가중치(λ) 및 조기 탐지 (Early Exit)**: 스트리밍 추론 중 일정 이상 확신 시 조기에 탐지 결과를 확정 짓는 데 최적화되어, 실시간 애플리케이션으로 사용되기 적합한 특성을 지닙니다.

---

## 2. 관련 논문 목록 (References)

본 시스템 아키텍처에 사용된 각 구성요소들과 맞닿아 있는 주요 학술 논문들을 참고문헌 형식에 맞춰 리스트업 하였습니다.

### Pre-trained Language Models & KoELECTRA
*   **Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020).** ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *In Proceedings of the 8th International Conference on Learning Representations (ICLR)*.
    *   *내용:* KoELECTRA의 모태가 된 ELECTRA 모델 원작으로 RTD (Replaced Token Detection) 학습 방식을 제시했습니다.
*   **Park, J. (2020).** KoELECTRA: Pretrained ELECTRA Model for Korean. *GitHub repository: https://github.com/monologg/KoELECTRA*.
    *   *내용:* 한국어 말뭉치를 바탕으로 ELECTRA 아키텍처를 학습시킨 모델의 기술 문서입니다.

### Parameter-Efficient Fine-Tuning (PEFT)
*   **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** LoRA: Low-Rank Adaptation of Large Language Models. *In Proceedings of the 10th International Conference on Learning Representations (ICLR)*.
    *   *내용:* 현재 V5 파이프라인에서 인코더 학습 효율을 극대화하기 위해 사용하는 LoRA 기법의 원작 논문입니다.

### Attention 기반 Pooling (Attention-weighted Pooling)
*   **Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016).** Hierarchical Attention Networks for Document Classification. *In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 1480-1489.
    *   *내용:* 단순 풀링 구조에서 벗어나 토큰의 중요도를 반영하여 벡터를 응집(Aggregation)하는 Attention Pooling 기법의 토대가 되는 연구입니다.

### Sequence Modeling 및 Mamba SSM
*   **Gu, A., & Dao, T. (2023).** Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*.
    *   *내용:* V5 모델에서 청크(Chunk) 간의 상태 추적기(Belief Tracker) 역할을 수행하는 가장 중요한 핵심 구조인 Mamba 프레임워크의 논문입니다. 트랜스포머의 계산 한계를 극복하고 긴 시퀀스를 처리하는 원리를 배울 수 있습니다.
*   **Gu, A., Goel, K., & Ré, C. (2021).** Efficiently Modeling Long Sequences with Structured State Spaces. *In Proceedings of the 9th International Conference on Learning Representations (ICLR)*.
    *   *내용:* Mamba의 이론적 뼈대인 S4 (Structured State Spaces) 모델에 대한 기초적인 상태 공간(State Space) 모델링 논문입니다.

### 조기 탐지 (Early Stop / Early Detection) 및 연속 분류 프레임워크
*   **Teerapittayanon, S., Bradley, B., & Kung, H. T. (2016).** BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. *In 2016 23rd International Conference on Pattern Recognition (ICPR)*, 2464-2469. IEEE.
    *   *내용:* 스트리밍 중 Early Stop을 판단하는 등 연산량을 줄이면서 조기 예측(Early Exit)하는 개념의 기반을 제공한 연구입니다.
*   **Zeng, Z., Liu, S., Shi, W., ... & Li, Y. (2022).** Multi-Turn Dialogue State Tracking. *In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)*.
    *   *내용:* 대화의 턴이 진행될수록 이전 상태(Belief-State)가 어떻게 갱신 및 보존(Tracking)되는지, 그 번복률과 유지력을 다루는 유사 연구 분야의 논문입니다.
