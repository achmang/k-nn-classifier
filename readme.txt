The purpose of this script is to illustrate how a k - nearest neighbour classifier works.

run this script with 'py k-nn-skin-class.py r g b k'

r is red value
g is green value
b is blue value
k determines quantity of neighbours

This classifier determines if an RGB value is a skin cell sample.
It makes this decision using existing knowledge of skin cells.

I have a dataset of skin samples and non-skin samples, and some RGB input value.
All these values are plotted on a 3dimensional space, using RGB as each axis.

The program will calculate the distance between the RGB input, and all the existing data on the graph.
K represents the number of closest neighbours.

If k = 3. It will calculate the 3 closest neighbours.

The classifier will determine if the RGB input is a skin cell, or not a skin cell
based on the higher amount of neighbours in the respected category.

So if K = 3, and 2 of its closest neighbours are non-skin samples. The classifier will
determine that the input value is not a skin cell.
