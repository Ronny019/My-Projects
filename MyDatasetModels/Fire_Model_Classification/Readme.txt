This project produces a machine learning classifier that will predict fire type (Fire.type) from the data provided.
The features provided are as follows:

Fire.type
Fire type was observed and classified as either:
-	surface fire (S): a fire burning in the litter on the forest floor, but not in the tree crowns
-	passive crown fire (PC): surface fire ignites tree crowns, individually or in groups, but fire does not transition from tree to tree
-	active crown fire (AC): surface fire ignites tree crowns and fire moves as a ‘wall of flame’ from tree to tree
See https://www.nwcg.gov/publications/pms437/crown-fire/active-crown-fire-behavior for detailed descriptions

Plot
A nominal identifier to distinguish between plots at each experimental site.

CBH
Canopy base height (m). Theory suggests that forest stands with higher CBH require higher fire danger conditions (drier, windier) to achieve crown fire (passive or active).

FMC
Foliar moisture content (% water content). Lower FMC is believed to be associated with higher flammability.

SH
Stand height, m. Mean height of dominant trees in a plot.

ws
Wind speed (km/h). Mean wind speed during the experimental burn. 

FFMC
Fine Fuel Moisture Code (unitless). Index of the estimated moisture content of the fine dead fuel in a closed conifer forest. Higher values represent lower moisture; maximum value is 101.

DMC
Duff Moisture Code (unitless). Index of the estimated moisture content of the decaying organic layers (fermentation and humus layers) in a conifer forest floor. Higher values represent lower moisture; no maximum value. 

DC
Drought Code (unitless). Index of the estimated moisture content of deepest organic layers in the soil and of the largest dead logs in a conifer forest. Higher values represent lower moisture; no maximum value. 

MC.SA
Estimated stand-adjusted dead litter moisture content (% water content), based on FFMC, DMC, stand type, season, and stand density (not shown). 