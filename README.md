
# RefineGAN

----------
This repository holds the code for RefineGAN, 

It is developed for research purposes only and not for commercialization. 
If you use it, please refer to our work. 

    @ARTICLE{8327637, 
    author={T. M. Quan and T. Nguyen-Duc and W. K. Jeong}, 
    journal={IEEE Transactions on Medical Imaging}, 
    title={Compressed Sensing MRI Reconstruction using a Generative Adversarial Network with a Cyclic Loss}, 
    year={2018}, 
    volume={PP}, 
    number={99}, 
    pages={1-1}, 
    keywords={Databases;Gallium nitride;Image quality;Image reconstruction;Machine learning;Magnetic resonance imaging;Training;Compressed Sensing;CycleGAN;DiscoGAN;GAN;MRI}, 
    doi={10.1109/TMI.2018.2820120}, 
    ISSN={0278-0062}, 
    month={},}
----------
Directory structure of data:

     tree data
     data/
    ├── brain
    │   ├── db_train
    │   └── db_valid
    ├── knees
    │   ├── db_train
    │   └── db_valid
    └── mask
        ├── cartes
        │   ├── mask_1
        │   ├── mask_2
        │   ├── ...
        │   └── mask_9
        ├── gauss
        │   ├── mask_1
        │   ├── mask_2
        │   ├── ...
        │   └── mask_9
        ├── radial
        │   ├── mask_1
        │   ├── mask_2
        │   ├── ...
        │   └── mask_9
        └── spiral
            ├── mask_1
            ├── mask_2
            ├── ...
            └── mask_9

    
    
Brain data is used for magnitude-value experiment, it is extracted from IXI dataset
Knees data is used for complex-value experiment, it is extracted from IXI dataset

----------


To begin, the template for such an experiment  is provided in `exp_dset_RefineGAN_mask_strategy_rate.py`

For example, if you want to run the training and testing for case knees data, mask radial 10%, please make a soft link to the experiment name, like this

    ln -s exp_dset_RefineGAN_mask_strategy_rate.py \
		  exp_knees_RefineGAN_mask_radial_1.py

To train the model

    python exp_knees_RefineGAN_mask_radial_1.py  \
		    --gpu='0'							 \
		    --imageDir='data/knees/db_train/'    \
		    --labelDir='data/knees/db_train/'    \
		    --maskDir='data/mask/radial/mask_1/' 
		    
Checkpoint of training will be save to directory `train_log`

To test the model

     python exp_knees_RefineGAN_mask_radial_1.py \
		    --gpu='0' 							 \
		    --imageDir='data/knees/db_valid/' 	 \
		    --labelDir='data/knees/db_valid/' 	 \
		    --maskDir='data/mask/cartes/mask_1/' \
		    --sample='result/exp_knees_RefineGAN_mask_radial_1/' \
		    --load='train_log/exp_knees_RefineGAN_mask_radial_1/model-20000.data-00000-of-00001'   


----------
The authors would like to thank Dr. Yoonho Nam for the helpful discussion and MRI data, and Yuxin Wu for the help on Tensorpack.


