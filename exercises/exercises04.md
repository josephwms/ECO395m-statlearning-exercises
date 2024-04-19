## Clustering and PCA

![](exercises04_files/figure-markdown_strict/unnamed-chunk-1-1.png)![](exercises04_files/figure-markdown_strict/unnamed-chunk-1-2.png)

## Market segmentation

*Use the data to come up with some interesting, well-supported insights
about the audience and give your client some insight as to how they
might position their brand to maximally appeal to each market segment.*

To get a basic idea of our data lets start with a two-way correlation
plot.

![](exercises04_files/figure-markdown_strict/unnamed-chunk-2-1.png)

Great! So already we’re seeing some clusters. I notice see two way
correlation within the following groups: \* group1\_familyvalues:
`parenting`, `religion`, `sports_fandom`, `food`, `school`, `family` \*
group2\_collegeboy: `college_uni`, `online_gaming`, `sports_playing` \*
group3\_fashionable: `beauty`, `cooking`, `fashion` \* group4\_yuppie:
`personal_fitness`, `health_nutrition`, `outdoors` \* group5\_neolib:
`politics`, `travel`, `computers` \* group6\_socialyte: `shopping`,
`chatter`, `photo_sharing`

Lets organize counts to see how many followers had at least two or more
tweets in at least 2, (or 3 for family\_values and socialyte) variables
of each cluster. Lets also filter out users who are in more than 3 of
these respective groups to eliminate generalists and get a better
personality portrait of our followers.

![](exercises04_files/figure-markdown_strict/unnamed-chunk-3-1.png)

Over 1500 users are in the loosely constructed \`familyvalues’,
‘yuppie’, and ‘socialyte’ groups, respectively. Even with filtering
efforts, however, many users are likely counted 2 or even 3 times.
Before we go farther in this direction, lets shift to machine learning
algorithms so that our clusters are definite and exhaustive. We’ll start
with k-means and k\_means++ clustering with k=5-7 clusters, since we see
6 key groups off the bat.

    ##   Cluster Users
    ## 1       1   740
    ## 2       2   670
    ## 3       3  4172
    ## 4       4   870
    ## 5       5  1430

![](exercises04_files/figure-markdown_strict/unnamed-chunk-4-1.png)

    ##   chatter current_events travel photo_sharing uncategorized tv_film
    ## 1   -0.02          -0.07  -0.21         -0.15         -0.09   -0.04
    ## 2   -0.04           0.12  -0.10         -0.03         -0.07   -0.01
    ## 3    0.04           0.11   1.76         -0.06         -0.04    0.08
    ## 4    0.16           0.20  -0.04          1.25          0.52    0.00
    ## 5    0.02          -0.02   0.00          0.05          0.11    0.44
    ## 6    0.00           0.03  -0.15          0.00          0.16   -0.05
    ##   sports_fandom politics  food family home_and_garden music  news online_gaming
    ## 1         -0.29    -0.26 -0.35  -0.26           -0.11 -0.12 -0.24         -0.23
    ## 2          2.00    -0.20  1.79   1.44            0.17  0.05 -0.08         -0.08
    ## 3          0.19     2.37  0.02   0.04            0.12 -0.04  1.96         -0.14
    ## 4         -0.19    -0.11 -0.18   0.04            0.16  0.57 -0.07         -0.05
    ## 5         -0.12    -0.17 -0.07   0.19            0.13  0.33 -0.19          3.08
    ## 6         -0.19    -0.18  0.41  -0.08            0.17  0.05 -0.05         -0.14
    ##   shopping health_nutrition college_uni sports_playing cooking   eco computers
    ## 1    -0.06            -0.33       -0.22          -0.23   -0.33 -0.16     -0.23
    ## 2     0.05            -0.16       -0.12           0.10   -0.12  0.20      0.07
    ## 3    -0.01            -0.21       -0.08          -0.01   -0.22  0.11      1.55
    ## 4     0.37            -0.06        0.00           0.18    2.58  0.08      0.07
    ## 5    -0.02            -0.18        3.08           2.00   -0.15 -0.03     -0.06
    ## 6     0.06             2.08       -0.21          -0.03    0.37  0.53     -0.07
    ##   business outdoors crafts automotive   art religion beauty parenting dating
    ## 1    -0.12    -0.32  -0.19      -0.18 -0.06    -0.30  -0.26     -0.31  -0.09
    ## 2     0.12    -0.08   0.70       0.17  0.08     2.18   0.29      2.07   0.04
    ## 3     0.36     0.11   0.15       1.11  0.00    -0.03  -0.17      0.02   0.20
    ## 4     0.27     0.03   0.15       0.06  0.13    -0.12   2.40     -0.07   0.16
    ## 5     0.03    -0.09   0.11       0.05  0.31    -0.13  -0.20     -0.15   0.02
    ## 6     0.07     1.61   0.09      -0.12  0.01    -0.17  -0.21     -0.10   0.18
    ##   school personal_fitness fashion small_business  spam adult
    ## 1  -0.25            -0.34   -0.26          -0.10  0.00  0.01
    ## 2   1.63            -0.11    0.01           0.10 -0.01  0.00
    ## 3  -0.04            -0.19   -0.18           0.24 -0.01 -0.09
    ## 4   0.20            -0.05    2.51           0.27 -0.04  0.02
    ## 5  -0.20            -0.18   -0.05           0.22  0.03  0.03
    ## 6  -0.14             2.05   -0.11          -0.06  0.00  0.01
    ##   Cluster Users
    ## 1       1  4532
    ## 2       2   759
    ## 3       3   682
    ## 4       4   570
    ## 5       5   440
    ## 6       6   899

![](exercises04_files/figure-markdown_strict/unnamed-chunk-4-2.png)

    ##   Cluster Users
    ## 1       1   517
    ## 2       2   371
    ## 3       3   618
    ## 4       4   805
    ## 5       5  1281
    ## 6       6   707
    ## 7       7  3583

![](exercises04_files/figure-markdown_strict/unnamed-chunk-4-3.png)

We are able to confirm using unsupervised learning pretty much the same
clusters we identified using intuition and basic tools. Labeled
variables have centroid values over a certain, relatively-high
threshold. Our results use standard K-means but we’ve verified that
results are repeatable with K-means++ start up.

As to selection of K we notice the following points: - Each K value
identifies a ‘spam’ category where no variables are dominant - K=5
splits up our ‘fashionable’ and ‘socialyte’ clusters between the four
non-spam groups, so that the only major feature of one of the clusters
is `photo_sharing`.  
- K=6,7 recovers the ‘fashionable’ cluster, and some of the ‘socialyte’
cluster. Its likely that the ‘socialyte’ characteristics `shopping`,
`chatter` and `photo sharing` represent popular uses of Twitter which
are more easily distributed between users in other categories.

Altogether, its clear that K=6 weeds out the most spam posts and
maintains an even distribution between the other categories, a key
requirement of clustering. Most importantly, it aligns with our
intuition. Our findings will rely on K-means clustering using K=6 and
identify five distinct non-spam groups.

#### Insights

We’ve identified 5 roughly even-sized market segments. Here are the two
largest in descending order: - **Health-conscious adults (likely
mid-twenties to thirties)**. In my city we call these yoga enthusiasts
and REI shoppers ‘yuppies’, short for ‘young professionals’. This market
segment shared twitter engagement in `health_nutrition`,
`personal_fitness`, `outdoors`. While not included in our graphic. This
segment also showed notable interest for `eco`. We recommend an approach
that shows sustainability efforts, and connects your product to outdoor
engagmeent and mental health. - **Traditional Americans**. Don’t forget
about the heartland, the silent majority. Your minivan moms and sports
bar dads. This market segment showed over-threshold engagement with more
categories than any other group, by far, indicating they are ‘classic
American consumers,’ not part of any niche group. key characteristics
align with traditional values, and include `family`, `food`, `religion`,
`sports fandom`, `school`, `parenting`. To appeal to this group, show
that your product could easily find its way to a children’s soccer game
or family reunion.

To appeal to both groups and maximize market outreach,perhaps you are
the beverage of choice for the modern parent… but not *too* modern.
Perhaps its being consumed on a good ol’ fashioned camping trip. Don’t
forget to put ice in the cooler!

## Association rules for grocery purchases

## Image classification with neural networks

In this problem, you will train a neural network to classify satellite
images. In the
[data/EuroSAT\_RGB](https://github.com/jgscott/STA380/tree/master/data/EuroSAT_RGB)
directory, you will find 11 subdirectories, each corresponding to a
different class of land or land use: e.g. industrial, crops, rivers,
forest, etc. Within each subdirectory, you will find examples in .jpg
format of each type. (Thus the name of the directory in which the image
lives is the class label.)

Your job is to set up a neural network that can classify the images as
accurately as possible. Use an 80/20 train test split. Summarize your
model and its accuracy in any way you see fit, but make you include *at
a minimum* the following elements:

-   overall test-set accuracy, measured however you think is
    appropriate  
-   show some of the example images from the test set, together with
    your model’s predicted classes.
-   a confusion matrix showing the performance of the model on the set
    test, i.e. a table that cross-tabulates each test set example by
    (actual class, predicted class).

I strongly recommend the use of PyTorch in a Jupyter notebook for this
problem; look into PyTorch’s `ImageFolder` data set class, which will
streamline things considerably. I’ll give you the first block of code in
my Jupyter notebook, which looks like this. I’ve handled the resizing
and normalization of the images for you – you can take it from here.

    # Necessary Imports
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    import matplotlib.pyplot as plt
    import numpy as np

    # Set the directory where your data is stored
    data_dir = '../data/EuroSAT_RGB'

    # Set the batch size for training and testing
    batch_size = 4

    # Define a transformation to apply to the images
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # Resize images to 32x32
         transforms.ToTensor(),  # Convert image to PyTorch Tensor data type
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize the images

    # Load the training data
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Create data loaders for training and testing datasets
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Print some samples to verify the data loading
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    print(images.shape, labels.shape)

    # Function to show an image
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

     # Get some random training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    # Show images
    imshow(torchvision.utils.make_grid(images))

    # Print labels
    print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(batch_size)))

One tip: in our example of a convolutional neural network in class, we
had black and white images, and therefore *one* input channel in our 2D
convolutions. These are RGB images here, and so you’ll need to modify
the first convolutional layer accordingly to handle *three* input
channels.
