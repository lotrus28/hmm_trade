# hmm_trade
Using Gaussian Mixrture Models to predict pricee movement

This is an attempt to make a price predictor bot.
As you can test: it works poorly. So far, I have managed to kick it into making 0.997 ROI per a trade, which is awful considering it recommends to take 40+ trades a month.
And that is the best performing coin. Other coins give even worse results. So if you hope this script can help you make money: it won't.

But it may, if you tweak it.
I suggest that you start with:
1. Measuring dependence on training data set: train in bear, test in bull, vice versa etc. When I was writing it, I was using Autumn+December data for training and testing on January. Perhaps that is the root of all inaccuracy;
2. Implement multiple step HMM: the transition matrix is 2D, and I have never figured out the way to make it 3D or more;
3. Implement initial parameter estimation. It is non-existent here. I used to try 4-8 number of states and take equiprobable transitions and sample average emissions, but there should be a better way to do that.
4. Find a way to replace Gaussians with other distributions. I strongly suspect, they may not be good at all for this kind of problem.