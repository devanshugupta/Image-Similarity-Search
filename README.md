# How to run
- The code is dependent so you will have to run all tasks sequentially to get the relevent data files.
- Run feature extraction .ipynb file to get the features, it may take a long time to run.
- Do make sure that the Caltech101 dataset 'download' parameter is set to True
- Connect you google drive and upload the files in case you want to run code on Google Colab.
- The path names for files may have to be changed as per your google drive.

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

  ## Results

Results for Probabilistic Relevance feedback :

Enter the query image id: 1
Enter no. of images t: 10
![sSbPeA9j76jAD4NyC0lDQnibLheU3CtJopW155ix0COCgLsGwjgOoQzz4lNQBiWyX_n9uKEpJeQwslfmdVsbGCeJt1ca-TL6wUWPM2AwHVaJLaxwbJpQ5JzByI_b53BSjqHJbwlMYbTzSq5PYJ5GbzA](https://github.com/devanshugupta/ImageSearch-CSE-515/assets/22978896/bd740bb1-debd-4821-9f79-8fafc68d167e)

￼
Feedback relevancy: 2 for Very relevant | 1 for relevant | -1 for irrelevant | -2 for very irrelevant | 0 to skip->
![-5jnraTpA1C6BnIYTIa3F8_eUpJBT4XZUZkAQAzSxEtJg0RGA1FX5gHnYlroiSDgjqnvaTiVcDwd40Qj1kreqjNAggYCf1MkfUi6Mt92hmJOkMbk_uwZohIJUOT_gHWk8aAjch79uacnMYNmFHs7U64](https://github.com/devanshugupta/ImageSearch-CSE-515/assets/22978896/311b0215-eae4-4f4c-8269-bed7bbd687e9)

￼
Enter the relevancy of 384: 1
Enter the relevancy of 36: 1
Enter the relevancy of 218: 1
Enter the relevancy of 6: 2
Enter the relevancy of 432: 1
Enter the relevancy of 430: 1
Enter the relevancy of 28: 1
Enter the relevancy of 400: -1
Enter the relevancy of 404: -1
Enter the relevancy of 350: -2
![AWL10gMJq64WUDAIfBk_hkcG4tqxrH8wTHO8YaKZxlWZEHunGeaLWQR5gi1AlW2UGe5PHCiF4I-HYwlaw6tDzvO86M7i1g1tjwSw_wKqD9VDjA2lmH2cNb_jXCqpG513Xk5dYODrAqyQoeDUjdPObsg](https://github.com/devanshugupta/ImageSearch-CSE-515/assets/22978896/8459b9c4-1826-4d8f-9e91-dc10af0c3a31)

￼
Do you want to continue giving feedback[Y/N]y
Enter the Number of images you want to view to give feedback:10
Feedback relevancy: 2 for Very relevant | 1 for relevant | -1 for irrelevant | -2 for very irrelevant | 0 if no feedback for an image ->
Enter the relevancy of 6: 2
Enter the relevancy of 218: 1
Enter the relevancy of 432: 1
Enter the relevancy of 384: 1
Enter the relevancy of 18: 2
Enter the relevancy of 36: 1
Enter the relevancy of 430: -1
Enter the relevancy of 28: 1
Enter the relevancy of 206: 1
Enter the relevancy of 12: 2
![xN4xFsk3MrBj7KdLd6TxH0GGIV-ZEYuz1OqqVjyRD7Xn2uG9MK6dn7wst5gsew6xEegjDsMwW39yRUJTq3dIF1fzaMREu1xSaiCaAGASngClYY--dHhNaCzvVqryZhVm6Hd3op0KtisrkBlA3gHKLXI](https://github.com/devanshugupta/ImageSearch-CSE-515/assets/22978896/c98131c0-bdfd-45d4-a229-81072bc92730)

￼
Do you want to continue giving feedback[Y/N]n
Do you have another query?[Y/N]n
