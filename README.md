Illinois Datathon 2024
Team : Fear of God

IVR Vision Statement – Quickly identify who is calling and why, and route them to the destination that can best resolve their inquiry

# Problem Statement

In this challenge, participants are tasked with analyzing the relations between call attributes and customer call reasons within the IVR (Interactive Voice Response) system with an emphasis on calls that the IVR could not handle (i.e., floor calls). Your objective is to delve into the provided dataset and use the attributes* to uncover insights:
 - Digitally Active (eservice_ind = 1)
 - Number of Accounts with Synchrony 
 - Account Open Date
 - Card Activation Status 
 - Account Balance
 - Account Status
 - Ebill Enrolled Status
 - Auto-Pay Enrolled Status
 - Delinquency History (Current vs. Past Due)

The goal is to complete the following:
 - Analyze and identify attribute associations to customer call reasons within the IVR system with an emphasis on calls that the IVR did not resolve (i.e., calls transferred to agent).
 - Understanding the attributes, their relationship to call reasons, and the IVR’s effectiveness to address the call reason, suggest opportunities to improve overall customer experience and satisfaction. We encourage creative approaches in data analysis, visualization, and predictive modeling to address this challenge.

Examples of what to improve: 
 - Adding functionality to the IVR
 - Preventing a re-call
 - Providing the caller a more effective channel-of-choice

# Divide into pieces

To find out the relationship between call attributes and customer call reasons, one need to know what are the call attributes and what are the possible customer call reasons. 

Initial thoughts: 
 - find out for each call reasons, the statistics of the attributes, better with visualizations 
 - supervised learning : do a regresssion models treating call reasons as labels and other attributes as covariances.
 - feature engineering and selection : what features are the most important?
        -- if manually selection, intuition like high credit results in more likely to run with payment 
        -- if auto selection, how to process the features
 - try different models to fit and compare with cross-validation, if time permits also with hyper parameters tuning
 - report on confusion matrix? if not 0-1 classification what would be a good deliverables?

Potential Aspects:
 sentimental scores from NLP? If voice is available? ...

 
# Other aspects of the project

Innovatives equipped with business knowledge are also important parts. To get a better understanding, one need to know

What are the all possible customer questions? I have some personal experiences, but maybe collect online? 
This can be regarded as unseen labels, how to deal with unseen labels?
 - How to generate unseen labels except by hand (is this necessary)?
 - How to predict calls and assign then with unseen labels? (classify failed calls into these classes) This can be a good
   insights to do

Have a better understanding of the current IVR system? 
my personal experiences includes IVR uncomprehensible, menus too long, not including the options I have, and? collect the complains 
from online and see what people says.


