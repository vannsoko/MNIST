# MNIST

<br /> <div align="center"> 

<h3 align="center">PyTorch MNIST Neuronal Network Classifier</h3>

<p align="center"> A project demonstrating how to train a Neuronal Network for MNIST digit recognition and optimize its hyperparameters using PyTorch and Optuna. <br />  </p> </div>

<!-- TABLE OF CONTENTS -->

<details> <summary>Table of Contents</summary> <ol> <li> <a href="#about-the-project">About The Project</a> <ul> <li><a href="#built-with">Built With</a></li> </ul> </li> <li> <a href="#getting-started">Getting Started</a></li> <li><a href="#usage">Usage</a></li><li><a href="#contributing">Contributing</a></li> <li><a href="#license">License</a></li> <li><a href="#contact">Contact</a></li> <li><a href="#acknowledgments">Acknowledgments</a></li> </ol> </details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project provides a comprehensive walkthrough for training a  Neural Network on the classic MNIST handwritten digit dataset. The entire workflow is self-contained in a Google Colab notebook for easy access and execution.

The primary focus is not only on building a model with **PyTorch** but also on leveraging **Optuna** to systematically and efficiently search for the best hyperparameters to maximize model accuracy. It serves as a practical template for applying automated hyperparameter tuning to deep learning projects.

### Built With

This project was built with the following technologies:

- Python
    
- PyTorch
    
- Optuna
    
- Google Colab
    

<!-- GETTING STARTED -->

## Getting Started

This project is designed to be run entirely in Google Colab, so there is no local setup required.
    

<!-- USAGE EXAMPLES -->

## Usage

After running the notebook, you will see the following process unfold:

1. **Dependency Installation**: The notebook will first install `pytorch` and `optuna`.
    
2. **Data Preparation**: The MNIST dataset is downloaded and prepared for training and validation.
    
3. **Optuna Study**: The hyperparameter search begins. Optuna will run multiple trials, each time training the CNN with a different set of hyperparameters (learning rate, optimizer, etc.).
    
4. **Live Results**: The output of each trial, including the validation accuracy, will be printed as the study progresses.
    
5. **Final Report**: At the end of the study, Optuna will report the best combination of hyperparameters found and the highest accuracy achieved.
    

This provides a hands-on example of how to automate the tedious process of hyperparameter tuning.

    



## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
    
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
    
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
    
4. Push to the Branch (`git push origin feature/AmazingFeature`)
    
5. Open a Pull Request
    

<!-- LICENSE -->

## License

This project does not have a license. Feel free to fork, modify, and use it as you see fit.

<!-- CONTACT -->

## Contact

vannsoko 

Project Link: [https://github.com/vannsoko/MNIST](https://github.com/vannsoko/MNIST "null")
