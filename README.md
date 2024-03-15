Spider-like Robot Final Project

Background:

  Genotype graph: a main body with 4 recursive upper legs and each recursive upper leg has recursive lower legs (see the following figure).
<img width="1069" alt="Screenshot 2024-03-15 at 12 53 08 AM" src="https://github.com/xinranlii/A_Life/assets/87915617/4084164c-9ad9-42ee-8c4a-7c0ba8a11cce">

Goal:

  To make the spider-like robot to move stabilized.
  
Process:

In hw3, to ensure symmetry of the model, the number of layers (every 4 upper legs is one layer) of upper leg and the number of layers of lower legs are randomized separately.
The fitness function is based on the moved distance of the center of the body. The conculsion is when the number of lower leg layers is equal or greater than the number of upper leg layers, the fitness score is better. 

In hw4, the mutation function randomly modifies the mass of upper legs and lower legs separetely, gear, the range of upper and lower legs' hinges separetely. 

In the final report, I use several regression models to distiguish and evaluate the efficiency on the modification of different features on the creature.

The final report uses two different fitness functions:

Generation 1: The fitness function is based on distance moved, the mutation function is same as the hw4. After running the regression models, the result shows that the variables I chose to modified does not affect the fitness score significantly. 

Generation 2: Based on the observation of generation 1, I modify the fitness function based on both distance moved and stabiliity for generation 2. The result shows that the modified fitness function better evaluates the creatures than the previous one. In another word, it is able to better represent detailed changes on the creatures.

Generation 3: I modified the mutation function to make changes in gears of the hinges in the upper legs and lower legs separately, where in the previous generations, same value was applied to all of them. As a result, the linear regression shows that the number of the upper-leg layers inversely and significantly affects the fitness score while the number of the lower-leg layes positively affects the fitness score.

Generation 4: I increased the range that generates the layer number of the lower legs, and decrease the range for the upper legs. The number of lower-leg layes is guaranteed to be greater than it of the upper-leg layers. I also increased the random range of mass for upper legs, and decrease it slightly for the lower legs. The result was not affected much in this generation.

Please check the report.txt for details.
