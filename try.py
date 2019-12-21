from  tensorgroup.models.networks.ResnetKeypointBuilder import ResnetKeypointBuilder as RKB
from tensorgroup.models.networks.BagnetBuilder import BagnetBuilder as BB

image_shape = (512,512,3) #(224, 224, 3)
#modelP = RKB.build_pose_resnet_50(input_shape = image_shape, num_outputs = 10)
modelB = BB.build_bagnet_9(input_shape=image_shape, num_outputs=10)
modelB.summary()
