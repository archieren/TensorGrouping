from  tensorgroup.models.networks.ResnetKeypointBuilder import ResnetKeypointBuilder as RKB
image_shape = (512,512,3) #(224, 224, 3)
model = RKB.build_pose_resnet_50(input_shape = image_shape, num_outputs = 10)
model.summary()
