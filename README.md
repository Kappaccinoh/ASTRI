# ASTRI HK (Applied Science and Technology Research Institute - Hong Kong) Jul - Aug '24

## Table of Contents
1. Introduction
2. Description of Tasking
3. Difficulties of Tasking
4. Subtask 1: Traffic Notice Parser (Llama3)
5. Difficulties of Subtask 1
6. Recommendations for future work
7. Subtask 2: Traffic Density Plotting System (Yolov10)
8. Summary
9. Acknowledgements

### 1. Introduction
#### ASTRI
Hong Kong Applied Science and Technology Research Institute Company Limited (ASTRI) was founded by the Government of the Hong Kong SAR in 2000 with the mission of enhancing Hong Kong’s competitiveness in technology-based industries through applied research. ASTRI’s R&D strategic focus covers five areas of applications: Smart City, Financial Technologies (FinTech), Intelligent Manufacturing, Health Technologies, and Application Specific Integrated Circuits.

#### Communication Technologies
My internship and work deals with Smart Mobility Technologies (SMT) under the Communication Technologies (CT) department classified under the Smart City branch of ASTRI. This department looks at the integration of AI/ML technologies, along with hardware and various infrastructure networks to improve the performance of the road transport network in Hong Kong.

### 2. Tasking and Goals
The overall purpose and functionality of the system is to improve the network of Traffic Reports published by the department of Transport at https://www.td.gov.hk/en/traffic_notices/index.html. What is desired by ASTRI is to develop a overall system that is able to display the corresponding Traffic Cameras per traffic notice published by the Transport Department. This includes real time mapping of the affected areas of traffic on the map of Hong Kong, showing clearly the extent of congestion caused by a traffic notice. Traffic Notices do not just include accidents, but road works, temporary road closures, changes to bus services, and temporary imposed speed limits to name the more common types of traffic notices.

### 3. Difficulties of Tasking
The project tasking is of a significant scale involving many layers of complexity. A significant portion of the problems associated with the difficulty of this task are structural in nature, meaning that no amount of hacking by engineers will be able to provide a complete and 100% accurate solution to fit the requirements of the tasking.

#### Structural Problems
Problems with Hong Kong's Traffic Infrastucture include the sparity and variance in positioning of Cameras (some exist but are unable to see the street clearly) - this implies that not all Traffic Notices may be covered as the street in question is not adequately seen by any street camera. Traffic Notices published by the Transport Department are generally not of a standardised format, meaning that several traffic related measures of changes may be fitted into one Traffic Notice - this adds one layer of complexity to parsing Traffic Notices. Inconsistencies in Traffic Naming conventions and lack of information is another added layer of complexity. Traffic Notices may not include government registered street names, meaning that any quick look up is not possible, else they might include points of reference such as landmarks, buildings, bus stops and even lamp post numbers. This makes a simple solution of mapping street name to street location difficult as several databases are required to be able to account for the variance in location description.

#### Other Problems
The rest of the problems mostly concern the training and finetuning of the AI/ML models used in both subtasks. As these applications are rather niche, creating our own databases and training data to ensure that the AI/ML models accuracy was up to par took up a significant amount of time and manual labour.

### 4. Subtask 1: Traffic Notice Parser (Llama3)
The first subtask here tries to tackle the initial problem of mapping traffic notices to street cameras. The rough plan of action of the system is described in the following steps (refer to the github attached for more information on the specific implementation):

Step 1. Parse traffic notices in HTML using a language model (Llama3) to extract the required streets in question

Step 2. Format the streets and call its formatted names through the OverpassAPI (OpenStreetMaps)

Step 3. Parse through all returned nodes from the call to find intersections/junctions/street names

Step 4. In the case that no junctions are present (either the traffic notice does not contain a specific mappable junction, or that street junction does not exist on OpenStreetMaps), take the average lat and lon of each specific node and return those coordinates, else return the specific junction’s coordinates

Step 5. Use the retrieved coordinates to parse through the list of known street cameras and pull up the closest coordinate and return the footage of that Street Camera

### 5. Difficulties of Subtask 1
The approach described was built and achieved an accuracy of over 90% of only one specific type of Traffic Notice ("Notices of Clearways") - which dealt with highway related traffic notices. The reason for this success and accuracy was because highways are well documented as "major road segments", and will almost 90% of the time be able to have a documented junction and corresponding street camera. However, this approach was ultimately discontinued when other types of traffic notices were fed into the system. These other types of traffic notices did not perform well on Llama3 and required more training on those specific formats, those which included additional information that OpenStreetMaps did not provide, such as bustops, buildings and lamp post IDs.

### 6. Recommendations for future work
The recommended plan of action to continue this system would require more effort invested into training Llama3 models, one for each type of traffic notice, and one in English and one in Traditional Chinese. This redundancy will increase the likelihood of finding the required street name due to double searching in two different languages.

The training of the respective Llama3 models would require searching through historical data published on Transport Department's website, and format it in a AI/ML friendly manner so that the models can be trained on them.

### 7. Subtask 2: Traffic Density Plotting System (Yolov10)
The second subtask here tries a different approach as the team lacked the manpower and time to create the needed datasets to train the Llama3 on. This subtask deals with mapping the extent of traffic density seen by street cameras and plotting them on a "heatmap" overlay of the map of Hong Kong. This approach may be called the backwards method, eliminating the uncertainty of Hong Kong street cameras, and starting with those cameras that the functional and provide good traffic coverage. The overall implementation of subtask 2 can be described in the following steps: 

Step 1: Start from all traffic cameras and retrieve thier live footage through the Traffic Department's open source API.

Step 2: Feed those images into a computer vision model (Yolov10) to detect density of traffic. We calculate the density of traffic based off "relative traffic density", which accounts for the variations of usual traffic due to peak and off peak hour, against the current reported number of vehicles at the time of reading. This is accomplished by first gathering the maximum number of observed vehicles every 15 minutes over the span of 1 week. The reported relative traffic density would be the number of currently seen vehicles as a percentage of the maximum number of observed vehicles.

Step 3: The relative traffic density is then partitioned into thirds, with the categorisation of "Low", "Medium" and "High" densities for each percentile respectively. 

Step 4: These categorical labels are then mapped onto the Hong Kong overlay being shaded into green, orange, and red respectively.

Step 5: For areas that are labelled "High" in traffic densiy are then returned. 

This subtask was considerably easier to manage compared to the initial parsing of traffic notices using a language model, and was completed in a span of 2 weeks divided among 2 individuals. Unfortunately, extensive testing and performance testing of this subtask was not done due to time constraints of this internship.

### 8. Summary
A partial system has been achieved providing the fundamental concept and base code for future endeavours. This tasking is definitely large in scale and considerable in complexity, and will require more focus and efforts to come up with a comprehensive system. More AI specialists will be needed to improve the performance of the LLM and CV models, and more scripting to generate datasets with ease.

### 9. Acknowledgements
Stella Wu (Head of CT/SMT)
George (Assistant Head of CT/SMT)
Aurora (Summer Intern)
Edith Lee (Summer Intern)
Edward Xiao (Summer Intern)
Lim Jia Wei (Summer Intern)