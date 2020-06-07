# teacher training
python teacher.py --arch wrn_40_2    --lr 0.05 --gpu-id 0
python teacher.py --arch wrn_40_1    --lr 0.05 --gpu-id 0
python teacher.py --arch wrn_16_2    --lr 0.05 --gpu-id 0
python teacher.py --arch vgg13       --lr 0.05 --gpu-id 0
python teacher.py --arch vgg8        --lr 0.05 --gpu-id 0
python teacher.py --arch resnet56    --lr 0.05 --gpu-id 0
python teacher.py --arch resnet20    --lr 0.05 --gpu-id 0
python teacher.py --arch resnet32x4  --lr 0.05 --gpu-id 0
python teacher.py --arch resnet8x4   --lr 0.05 --gpu-id 0
python teacher.py --arch ResNet50    --lr 0.05 --gpu-id 0
python teacher.py --arch ShuffleV1   --lr 0.01 --gpu-id 0
python teacher.py --arch ShuffleV2   --lr 0.01 --gpu-id 0
python teacher.py --arch MobileNetV2 --lr 0.01 --gpu-id 0


# student training

# similar-architecture
python student.py --t-path ./experiments/teacher_wrn_40_2_seed0/   --s-arch wrn_16_2     --lr 0.05 --gpu-id 0
python student.py --t-path ./experiments/teacher_wrn_40_2_seed0/   --s-arch wrn_40_1     --lr 0.05 --gpu-id 0
python student.py --t-path ./experiments/teacher_resnet56_seed0/   --s-arch resnet20     --lr 0.05 --gpu-id 0
python student.py --t-path ./experiments/teacher_resnet32x4_seed0/ --s-arch resnet8x4    --lr 0.05 --gpu-id 0
python student.py --t-path ./experiments/teacher_vgg13_seed0/      --s-arch vgg8         --lr 0.05 --gpu-id 0
# different-architecture
python student.py --t-path ./experiments/teacher_vgg13_seed0/      --s-arch MobileNetV2  --lr 0.01 --gpu-id 0
python student.py --t-path ./experiments/teacher_ResNet50_seed0/   --s-arch MobileNetV2  --lr 0.01 --gpu-id 0
python student.py --t-path ./experiments/teacher_ResNet50_seed0/   --s-arch vgg8         --lr 0.05 --gpu-id 0
python student.py --t-path ./experiments/teacher_resnet32x4_seed0/ --s-arch ShuffleV1    --lr 0.01 --gpu-id 0
python student.py --t-path ./experiments/teacher_resnet32x4_seed0/ --s-arch ShuffleV2    --lr 0.01 --gpu-id 0
python student.py --t-path ./experiments/teacher_wrn_40_2_seed0/   --s-arch ShuffleV1    --lr 0.01 --gpu-id 0

