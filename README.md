# Satellite_pose_estimation

Satellite 6 Degree of freedom pose estimation 

## Description

These are notebook for solving the 6D satellite pose estimation challenge proposed by the European Space Agency in the kelvin project cf. https://kelvins.esa.int/pose-estimation-challenge-post-mortem/home/

## Getting Started

### Dependencies

* Python modules requirements  : 
		torch, 
		torchvision, 
		pytorch-lightning, 
		numpy, 
		matplotlib, 
		pillow 

### Download data 

You can download SPEED Data from here : https://kelvins.esa.int/satellite-pose-estimation-challenge/data/

### Training model
 
* If you want to train locally : 
```
open ./local_runs/local_run_notebook.pynb
```
```
change DATA_PATH to the path of your local SPEED data directory after download, cf. #FIRST_TIME_SETUP comment 
```
* If you want use the azure ml training pipline : 
```
open azure_ml_run_notebook.pynb
```
```
creat an azure account 
```
```
change your DATA_PATH, subscription and tenant ids cf. #FIRST_TIME_SETUP comment 
```
```
if you want to costumize your working space, compute, dataset, environement, and experiment change the corresponding variables along the notebook 
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

Salem Ben el cadi 
[My linkedin](https://www.linkedin.com/in/salem-benelcadi-ba6a7398/)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the Apache License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)



 
