Paper: A Cotton Organ Segmentation Method with Phenotypic Measurements from a Point Cloud Using a Transformer

We created a dataset of three categories of cotton point cloud plants, including leaves, bolls, and branches. Each plant has a total of 40,960 dots.
The path of the data set is in./mydata
If you want to run this code, execute the following command: python train_partseg.py --model pointnet2_part_seg_msg
--normal --log_dir pointnet2_part_seg_msg
