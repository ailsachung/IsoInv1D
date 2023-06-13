# IsoInv

A python model to invert isochrone observations in deep polar ice sheets

# Files description

* agemodel.py: Main code which computes the age of the ice along a given radar profile.
* maps.py: Python script to draw the bedrock and surface map and overlay with radar lines


# How to run IsoInv?

* To run a single radar line (in terminal):

	`python age_model.py single_radar_line_directory/`

* To run a set of radar lines (in terminal):

	`python maps.py all_radar_lines_directory`

# What is the structure of the directory containing all radar lines?

* a directory per radar line
* parameters_all_radar_lines.yml: file containing parameters identical for all radar lines
* parameters-maps.yml: file containing parameters for the maps.py code.
* ages.txt: text file with a list of ages for the isochronal layers: column1=ages, column2=sigma  
* AICC2012.txt: age-depth profile of existing ice core with columns - depth(m), ice-extrapolated depth(m), ice-equivalent accumulation rate (m/yr), ice age (yr before 1950), age uncertainty(yr)

# What is the structure of a radar line directory?

* radar-data.txt: file containing the radar data for the radar line (see below for a description).
* parameters.yml (OPT): python file containing various parameters for the radar line. This file is optional since these parameters can be defined in the the directory upstream for all radar lines in the parameters_all_radar_lines.yml file.
* ages.txt (OPT): text file with a list of ages for the isochronal layers for individual radar lines: column1=ages, column2=sigma of ages. You can also define this file in the directory upstream for all radar lines.

# What is the output of age_model.py?

AgeModel.py creates a set of text files containing numerical outputs and pdf files containing graphs.

* a.txt, accumulation.pdf: Average accumulation rate along the radar line, as well as accumulation for each layer (as in Cavitte et al., TC, 2018).
* m.txt, m.pdf: Melting rate along the radar line.
* pprime.txt: pprime.pdf: parameter along the radar line.
* p.txt: p.pdf: parameter along the radar line.
* resi_sd.txt: resi_sd.pdf: reliability index (residual standard deviation) along radar line
* agebottom.txt: Various ages and vertical resolution at the bottom of the ice sheet along the radar line.
* ageisochrones.txt: Average ages of isochrones as calculated by the model.
* agehorizons.txt: Average ages of the non dated horizons as calculated by the model.
* twtt.txt: Two way travel time of a set of horizons dated by the model, to compare with radar data.
* res_max.txt: depth and maximum age according to age density threshold (< 20 kyr/m Oldest Ice target)
* stagnant.txt: stagnant ice thickness (if provided: basal unit and differnce between stagnant ice and basal unit thickness)
* [drill].txt: [drill].pdf (OPT): If a drill name and location was specified in the parameters.yml file, these files will be produced showing various parameters for the drill ice column profile
* Depths_[drill].pdf (OPT): isochrone depths and ages at drill location
* vertical_v_[drill].pdf (OPT): vertical velocity profile at drill location

* Data.pdf: Radar data along the profile (isochrones and bedrock).
* Model.pdf: Modelled age of the ice along the radar profile, as well as observed isochrones.
* Model-steady.pdf: Modelled steady age of the ice along the radar profile, as well as observed isochrones.
* Model-confidence-interval.pdf: Standard deviation of the modelled age along the radar profile.
* Temperature.pdf: Modelled temperature along the radar profile.
* Thinning.pdf: Modelled thinning function along the radar profile.
* AccumulationHistory.pdf: Layer per layer accumulation along the radar profile.
* AgeMisfit.pdf: Misfit between modelled and observed ages along the isochrones.
