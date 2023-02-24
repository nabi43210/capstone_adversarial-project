# adversarial example

�н��ý����� �������� �ǵ������� �߸��� ����� ������ ���. <br>
input �����Ϳ� �̼��� ��ȭ�� �־�, ����� �˱� ������� ������ �ý��ۿ����� ������ �ٸ� ��������� ���̵��� �����. <br>
�� ���������� �������� �����ϱ� ����� �̼��� input �������� ���� ���, adversarial exampe(���� AEs)�� ���� �ۼ��Ͽ���. 

## ����ϰ���
�̹����� gradient descent�� �����Ѵ�. <br>
�������� adversarial example�� ����� ����̴�.
<br>
�� ����� ����� ��ǥ���� ����� GoodFellow�� FGSM(The fast Gradient Sign Method)�̴�. ���� loss function�� ����� ��, �ش� �������� gradient�� sign function�� ����(�� ������ �������� �̵�)���� ����� �����Ѵ�. �ڼ��� ������ �Ʒ��ʿ� �ۼ��Ǿ� �ִ�. `���� �ڵ�� �ش� ���� �������� ��ũ�� �����Ͽ���.(�߰�����) `

### �� gradient + sign function
gradient descent(��� �ϰ���)�� �̺а����Լ��� local minimum�� ã�� ��ǥ���� �˰����̾���. <br>
�׷��� adversarial�� ������� ���Ǹ� input data�� `gradient������ ������`�� �߰��ϴ� �� ���δ�. gradient������ ����� �߰��� input�� �����׿��� loss�� �ּ�ȭ�ϴ� �н��ý����� ������Ų��.
<br> 
>���� ���� gradient descent�� �н� ���(�� ���� ���)�� ���⿴����, adversarial�� descent ���� ����� ���Ⱑ �ƴ� input�̴�.<br>
�� ������ ����н� �� $y=ax+b$ �� loss function�� ���Լ� $L'$�� ���<br>
${dL\over da} =2x(ax+b-y)$
�� �н� ����̾�����, ���� ���������� ����Ͽ� ������ x�� �����Ѵ�.

1. <b>input�� ���� loss function�� graadient����ϱ�</b><br>
$\nabla_xL(\theta,x,y)$ , <br>($\nabla$�� ������ �̺� ������), 
�̶� x�� $x��R^{w\times h\times c}$�� 3���� data input(�ʺ񡿳��̡�ä��) -> ���� �������� ��� �ʺ�x������ 2���� 2���� ������<br>

2. <b>gradient�� ������ ���ϱ�</b><br>
$sign(\nabla_xL(\theta,x,y))$

3. <b>Epsilon �߰�</b><br>
$\epsilon*sign(\nabla_xL(\theta,x,y))$<br>
($||x-\hat{x}||_�ġ�\epsilon$ �� �Ǵ� $\epsilon$�� ã�� ���Ѵ�. �� maximum perturbation, ���� �����Ϳ� ũ�� ������ ���� �����鼭 �ý����� �� ������Ű�� ������ ��� $\epsilon$ )<br>

4. <b> input data ����</b><br>
$x+\epsilon*sign(\nabla_xL(\theta,x,y))$
<br>
adversarial example �ϼ�



## adversarial patch
�̹��� �ϳ��ϳ����� ����ȭ�� adversarial example�� ����ϴ� ���� �ƴ϶�, adversarial patch�� ��ƼĿó�� ���̱⸸ �ص� ������ �ý������� �Ͽ��� �ν��� ��ư� ���� �� �ִ�. 
<br>
`image�� ������ �������� �ʾƵ� GAN�� Discriminator�� input���� ������ ���� patch�� �ٱ� ������, adversarial�� 2�� ������ ���� �� ���� ������ ����ȴ�.`



