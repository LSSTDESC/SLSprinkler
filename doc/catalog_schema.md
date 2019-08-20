# Catalog Schema for Sprinkler

## Necessary Inputs from Lensed Catalogs (OM10, Goldstein et al.)
### General Properties
* system_id
	* INT
	* Identifier in Catalog
### Lens Images Properties
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
* t_delay_img
	* List
	* Time Delays of each image (days)
* magnification_img
	* List
	* Image magnification

### Lens Galaxy Properties
* lens_sed
	* Str
	* SED filename of lens galaxy
* z_lens
	* Float
	* Redshift of lens galaxy
* reff_lens
	* Float
	* Effective Radius of lens galaxy
* ellip_lens
	* Float
	* Ellipticity of lens galaxy
* phie_lens
	* Float
	* Position Angle of lens galaxy
* av_lens
	* Float
	* Lens Galaxy AV
* rv_lens
	* Float
	* Lens Galaxy RV
