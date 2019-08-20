# Catalog Schema for Sprinkler

## Necessary Inputs from Lensed Catalogs (OM10, Goldstein et al.)
### Lens Images Properties
* system_id
	* INT
	* Identifier in Catalog
* n_img
	* INT
	* Number of Images in system
* z_src
	* List
	* Redshifts of Sources
* x_img
	* List
	* X locations of images relative to lens (arc seconds)
* y_img
	* List
	* Y locations of images relative to lens (arc seconds)
* t_delay
	* List
	* Time Delays of each image (days)
* magnification
	* List
	* Image magnification

### Lens Galaxy Properties
* lens SED
	* Str
	* SED filename of lens galaxy
* ZLENS
	* Float
	* Redshift of lens galaxy
* REFF
	* Float
	* Effective Radius of lens galaxy
* ELLIP
	* Float
	* Ellipticity of lens galaxy
* PHIE
	* Float
	* Position Angle of lens galaxy
* LENS_AV
	* Float
	* Lens Galaxy AV
* LENS_RV
	* Float
	* Lens Galaxy RV