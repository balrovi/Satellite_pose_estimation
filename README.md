# Satellite_pose_estimation

1) You can download SPEED Data from here : https://kelvins.esa.int/satellite-pose-estimation-challenge/data/

2) 

if you train locally : 
	
	open ./local_runs/local_run_notebook.pynb and change DATA_PATH to the path of your local SPEED data directory after download cf. #FIRST_TIME_SETUP comment 

if you use the azure ml training pipline : 
	open azure_ml_run_notebook.pynb
	creat an azure account 
	change your DATA_PATH, subscription and tenant ids cf. #FIRST_TIME_SETUP comment 
	if you want to costumize your working space, compute, dataset, environement, and experiment change the corresponding variables along the notebook 
 
