# Schema of the truth tables for the SLSprinkled lensed objects in the DDF

## Truth table types

DB name | Table names | Content
--- | --- | ---
`lens_truth.db` | `agn_lens`, `sne_lens` | information about the lensing mass, modeled as SIE + external shear, and the light profile of the lens galaxy, modeled as a Sersic bulge plus disk
`lensed_agn_truth.db` | `lensed_agn` | information about the lensed AGN images
`lensed_sne_truth.db` | `lensed_sne` | information about the lensed SNe images
`host_truth.db` | `agn_hosts`, `sne_hosts` | information about the host galaxy, modeled as Sersic bulge plus disk

## Reference catalogs

- SLSprinkler heavily uses the OM10 catalog of lensed quasars (Oguri and Marshall 2010) and the Goldstein et al 2018 catalog of lensed SNe. The lensing mass and source position are taken from these catalogs.
- The systems in the above catalogs are matched to a lens galaxy (light) and a host galaxy in DC2 via the Fundamental Plane relation.

### Schema for `lens_truth.db`

All the values are taken from OM10 unless otherwise indicated.

Quantity Label | Unit | Definition
--- | --- | ---
`lens_cat_sys_id` | - | Unique integer identifier for the lens system
`dc2_sys_id` | - | Unique integer identifier for the lens galaxy in cosmodc2
`vel_disp_lenscat` | km/s | velocity dispersion of the lens
`phie_lens` | degree | Orientation angle, with origin at the x-axis
`ellip_lens` | - | One minus the axis ratio (1 - minor/major) of the SIE lens
`ra_lens` | degree | Right ascension
`dec_lens` | degree | Declination
`redshift` | - | Cosmological redshift
`gamma_lenscat` | - | External shear modulus
`phig_lenscat` | degree | External shear angle (`0.5*arctan(shear_2_lenscat/shear_1_lenscat)`, with origin at the x-axis
`shear_1_lenscat` | - | External shear component 1
`shear_2_lenscat` | - | External shear component 2
`major_axis_lens` | arcsec | Major axis of lens light, from cosmodc2
`minor_axis_lens` | arcsec | Minor axis of lens light, from cosmodc2
`position_angle` | degree | Orientation angle of Sersic light (`arctan(e2/e1)`), from cosmodc2
`sindex_lens` | - | Sersic index, from cosmodc2
`phie_cosmodc2` | degree | Orientation angle of the Sersic light, with origin at the x-axis, from cosmodc2
`ellip_cosmodc2` | - | One minus the axis ratio (1 - minor/major) of the Sersic light, from cosmodc2
`shear_1_cosmodc2` | - | Weak lensing external shear component 1 at the lens position from cosmodc2
`shear_2_cosmodc2` | - | Weak lensing external shear component 2 at the lens position from cosmodc2
`kappa_cosmodc2` | - | Weak lensing external convergence at the lens position from cosmodc2
`av_mw` | - | Extinction in V-band, for galaxy light profile
`rv_mw` | - | Ratio of total to selective extinction in B and V bands, for galaxy light profile

### Schema for `lensed_agn_truth.db`

### Schema for `host_truth.db`

Please contact Ji Won Park @jiwoncpark with feedback or questions.