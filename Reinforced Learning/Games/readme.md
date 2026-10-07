# Dino 

# Snake 
original version was stuck at eating 3-4 apples, even after 7000 training episodes.

What makes it so difficult to automate via RL?
- First apple is pretty easy because it is always at the same position
- in post 1 apple, it becomes much more difficult:
  - every new apple appears in a random position -> complete new discovery phase each time
  - body of snake grows -> each game has a complete different setup
  - many random factors to factor in
 
I wanted only to rely on pure vision, so only on computers / robots see. So, no implementation of the position of the apple and hardcoded algorithms to go to the apple. The model has to figure it out itself

this version is an updated version of v.1.
Improvements are:
- added a 5th channel for snake-body detection
- implemented 'Multi-Steps-Return'
- implemented 'Prioritzed Experience Replay (PER)'
- implemented a 'boost' function -> gives snake an incentive to go find the next apple, so explore more. By this it knows there is a next one
