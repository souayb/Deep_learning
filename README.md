# DL_project
 
# Lab7:
    - In this segment I talk about how to increase the model accuracy by sharing the lattent representation of A Unet like  model with a simple classifer 
    1.  Models :
        - BASE:
            - Encoder part of the unet + global pooling + FC 
        - BASE + ENCODER:
            - Add Encoder out as skip connection 
        - FULL:
            - The classifer is a full Unet + ENCODER
        - FULL_SKIP:
            - FULL + ENCODER - DECODER out as skip connection
    2. Result:
        - BASE:
            - 55-56 % (cifar100)
            - 81-84 % (cifar10)
        - FULL : 
            - 60-61 % (cifar100)
            - 85-86.63 % (cifar10)
        <img src="Lab7/cifar10.png" width="500">
        <img src="Lab7/cifar100.png" width="500">
    3. Possible improvements:
        - Data augmentaton: This models are implemented without any data augmentaton
        - Try Two classifer: Ensemble of the Image segmentation and FULL_SKIP, where output are crossshared 
        - Hyperparameter tuning of Unet and extrapolate to the full model
    4. Issues:
        - The bigger models overfit easily with small tasks. 
        - running time:
            - BASE: 
            - FULL model : 240 second on average 
