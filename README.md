

# CSE515 Phase 3 
> experimenting with clustering, indexing and relevance feedback

This phase involves 5 tasks concerning the above areas.

## Execution Environment

### Environment

- Processor: Quad-core processor or higher
- Memory: 8 GB Ram or higher
- Operating Systems: Windows 10 or later, Mac OS 10 or later, Linux

### Libraries Installed

- Python >= 3.7
- NumPy >= 1.19.2
- torch>=1.11.0
- torchvision = 0.12.0
- sciPy = 1.5.2
- scikit-learn (distances)
- Pillow
- matplotlib
- PIP
- pandas
- Pickle
- CV2


## Execution Instructions

This phase involves code stored in the form of ipynb files. These files can be run on Jupyter notebook or the Google Colab.

- Task0a: After opening task0a.ipynb you have to run all the cells present in that file. The last cell will display the output of the program.

- Task0b: After opening task0b.ipynb you have to run all the cells present in that file. The last cell will display the output of the program.

- Task1: After opening task1.ipynb you have to run all the cells present in that file. The last cell will ask for input k which represents the number of latent semantics and after the input is given the program will display the output of the program.
- Task2: After opening task2.ipynb you have to run all the cells present in that file. The last cell will ask for input c which represents c most significant clusters and after the input is given the program will display the output of the program.
- Task3: After opening task3.ipynb you have to run all the cells present in that file. The last cell will display the output of the program.
- Task4a: After opening task4a.ipynb you have to run all the cells present in that file. The last cell will display the output of the program.
- Task4b: After opening task4b.ipynb you have to run all the cells present in that file. The last cell will display the output of the program.
- Task5: After opening task5.ipynb you have to run all the cells present in that file. The last cell will display the output of the program.


## Implementation

- Task 0a: Implements a program which computes and prints the “inherent dimensionality” associated with the even numbered Caltec101 images.
- Task 0b: Implement a program which computes and prints the “inherent dimensionality” associated with each unique
label of the even numbered Caltec101 images.
- Task 1: Implements a program which,
    – for each unique label l, computes the corresponding k latent semantics (of your choice) associated with the even numbered Caltec101 images, and
    – for the odd numbered images, predicts the most likely labels using distances/similarities computed under the label-specific latent semantics.
value.
- Task 2: Implements a program which,
    – for each unique label l, computes the corresponding c most significant clusters associated with the even numbered Caltec101 images (using DBScan algorithm); the resulting clusters are visualized both

    - as differently colored point clouds in a 2-dimensional MDS space, and
    - as groups of image thumbnails.
        and

    – for the odd numbered images, predicts the most likely labels using the c label-specific clusters.
- Task 3: Implements a program which,

    – given even-numbered Caltec101 images,

    - creates an m-NN classifer (for a user specified m),
    - creates a decision-tree classifier,
    - creates a PPR based clasifier.

    – Using the feature space of our choice, for the odd numbered images, it predicts the most likely labels using the user selected classifier.

- Task 4:

    – 4a: Implements a Locality Sensitive Hashing (LSH) tool (for Euclidean distance) which takes as input 
    
    (a) the number of layers, L, 

    (b) the number of hashes per layer, h, 

    (c) a set of vectors as input

    and creates an in-memory index structure containing the given set of vectors

    – 4b: Implements a similar image search algorithm using this index structure storing the even numbered Caltec101
    images and a visual model of our choice : for a
    given query image and integer t,

    - visulizes the t most similar images,
    - outputs the numbers of unique and overall number of images considered during the process

- Task 5: Considering the tag set “Very Relevant (R+)”, “Relevant (R)”, “Irrelevant (I)”, and “Very Irrelevant (I-)”, this program implements 

    – an SVM based relevance feedback system,

    – a probabilistic relevance feedback system 

    which enable the user to tag some of the results returned by 4b as and then return a new set of ranked results, relying on the feedback system selected by the user, either by revising the query or by re-ordering the existing results.
