

# Goals
- I created this file as a smaller more compressed way of describing what I have been doing + what I have been describing
as a replacement for testing documentation to describe how my system behavior is

- I recognized early on that that the costs of testing was really high as the evalluation of the dataset quickly became difficult 
to pin down. So opted to then create + bootstrap my own testing utilities to specifically interface and retrieve informataion in form 
As well as write reports on why i determine that my tools are done to do welll

This allows me to have a more easier way to showing my certainty, when sovling later problems that necessitate examination of my code structure
- i use actual testing if i use the computational power to actually examine the results and that the problems are wewll defiend


# Insights 
- I made careful consideration to examine the behavior of Nivetha's preprocessing pipeline to transform the format that they use into something usable to transform the datasets, I decided which functions in the code I decide to keep and what I decide to lose. Learning from last time issues with. With the dataset, I am able to visually


# Behavior of the CLI Interfacing Tools
- I used matplotlib to sample a couple of the npy datasets in my directory and verifed that for the sample, that the visualizer takes the images from the directory file and outputs it accordingly. I specificaally asked this question 

- given the directory configuraation - does LGE image print correctly? (Given a patient directory with a plotted image i see as ground truth, does the image visualizer corrrectly show the same image?)


- i sampled 2 elemnts and saw correctness --> I heuristically generalize my case as likely covering all the cases.
- more specifically i sampled patient 14163, slice 4 which I sampled with stochasticity. I saw that the images i plotted
 on matplotlib locally corresponded with the cine slilces showed on the visualizer tool 
--> Therefore it is highly likely that this tool does transform the visual lrepresentations correctly.







# Uncertainties the visualizer helps me address
- Does the swin model correctly, segment to a reasonalble degree the raw slices?
- is the collection of images that they collect associated with one another? --> Yes I visually verified it


# Behavior of the acquisition of the dataset + any nuances 
- It is infeasible to evaluate the quality of the dataset as that is too difficult for me to evaluate





#TODO: - Examine the structures of the dataset, size number of patients, model dtype size, (128x128)]


# 

Dataset Size - How I use this -
---> This is documentation on how to use + how it works?




# Insights about Dataset
- Orientational Features? asd
- Is the orientation of the SA slices the same orientaation? 
    - observed in the dataset that features that are orientation dependent are important
    - need information informing about the orientation? --> need insertion spot.


 
# Patient 12469 Evaluating considerations related to orientation
x - 100.0
y = 87.4
# Accesssing data related to orientation - 12534
- I can access the pieces of data related to orientation 
## Is the LV Insertion Data Valid? 
- yes, I heuristically tested it
- corrrecting the orientation.


## Hit
- i developed the tool for creaating it, it diretly ipmacted by reducingt the activation energy of being able to recognize the image orientation issue



# Ideas - Can I add interpretability to the ECG?
 - orietnation issue
- model - validation 
- interpretability - ECG 

# ECG
 - Interpretability



# Personal + Developmental Considerations made
- I made heavy use of a Makefile as a important tool for allowing the execuution of complex commands while
- I learaned how to use python package environments, docker containers (learning to export env variables + forward internal gui things) to the outside so that I can on my side be able to analyze and interface with data so that *I* can evaluate if the dataset is working asa intended + getting other bits of information (I can reduce information uncertainty of multiple variables)
- I am continuig to record + document insights that I find really relelvant to record for later through obsidian + retrieving

# Bootstrapping - I can show what I have been learning by showing Derek? 
- can i show the data structru
