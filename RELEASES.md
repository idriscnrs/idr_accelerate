# Releases

## Version 2.0.0
*August 2023*

This release is a complete rewrite of idr_accelerate.

#### New features

- idr_accelerate is now used as a launcher in lieu of `accelerate launch`.
- Now needs to be called in the distributed launcher (namely srun) so that each
rank creates its own file.
- Can now pass a config file to idr_accelerate, which will be merged with the
added option from this tool.


## Version 1.0.0
*May 2023*

First version of idr_accelerate. The script is used as a stand-alone script and
will create every configuration file accelerate might need. It requires a
bit of a trick to feed those files to accelerate correctly through SLURM.
