# adversarial example

학습시스템을 교란시켜 의도적으로 잘못된 결과를 만들어내는 방법. <br>
input 데이터에 미세한 변화를 주어, 사람은 알기 어렵지만 딥러닝 시스템에서는 완전히 다른 예측결과를 보이도록 만든다. <br>
이 문서에서는 육안으로 구별하기 어려운 미세한 input 데이터의 변형 방법, adversarial exampe(이하 AEs)에 대해 작성하였다. 

## 경사하강법
이미지에 gradient descent를 적용한다. <br>
고전적인 adversarial example을 만드는 방법이다.
<br>
이 방법이 적용된 대표적인 방법이 GoodFellow의 FGSM(The fast Gradient Sign Method)이다. 모델의 loss function을 계산한 후, 해당 데이터의 gradient에 sign function을 적용(즉 기울기의 방향으로 이동)시켜 노이즈를 생성한다. 자세한 원리는 아래쪽에 작성되어 있다. `구현 코드는 해당 문서 마지막에 링크로 삽입하였다.(추가예정) `

### ※ gradient + sign function
gradient descent(경사 하강법)은 미분가능함수의 local minimum을 찾는 대표적인 알고리즘이었다. <br>
그러나 adversarial의 방법에서 사용되면 input data에 `gradient방향의 노이즈`를 추가하는 데 쓰인다. gradient방향의 노이즈가 추가된 input은 딥러닝에서 loss를 최소화하는 학습시스템을 교란시킨다.
<br> 
>또한 기존 gradient descent의 학습 대상(즉 수정 대상)은 기울기였으나, adversarial의 descent 수정 대상은 기울기가 아닌 input이다.<br>
즉 기존의 기계학습 모델 $y=ax+b$ 의 loss function의 도함수 $L'$의 경우<br>
${dL\over da} =2x(ax+b-y)$
가 학습 대상이었으나, 모델이 고정적임을 고려하여 데이터 x를 수정한다.

1. <b>input에 따른 loss function의 graadient계산하기</b><br>
$\nabla_xL(\theta,x,y)$ , <br>($\nabla$는 다차원 미분 연산자), 
이때 x는 $x∈R^{w\times h\times c}$의 3차원 data input(너비×높이×채널) -> 단일 데이터의 경우 너비x높이의 2차원 2차원 데이터<br>

2. <b>gradient의 역방향 취하기</b><br>
$sign(\nabla_xL(\theta,x,y))$

3. <b>Epsilon 추가</b><br>
$\epsilon*sign(\nabla_xL(\theta,x,y))$<br>
($||x-\hat{x}||_∞≤\epsilon$ 이 되는 $\epsilon$을 찾아 곱한다. 즉 maximum perturbation, 원본 데이터에 크게 지장을 주지 않으면서 시스템을 잘 교란시키는 최적의 상수 $\epsilon$ )<br>

4. <b> input data 적용</b><br>
$x+\epsilon*sign(\nabla_xL(\theta,x,y))$
<br>
adversarial example 완성



## adversarial patch
이미지 하나하나마다 최적화된 adversarial example를 계산하는 것이 아니라, adversarial patch를 스티커처럼 붙이기만 해도 딥러닝 시스템으로 하여금 인식을 어렵게 만들 수 있다. 
<br>
`image에 변형이 가해지지 않아도 GAN의 Discriminator에 input으로 들어오는 순간 patch가 붙기 때문에, adversarial의 2차 방어막으로 쓰일 수 있을 것으로 예상된다.`



