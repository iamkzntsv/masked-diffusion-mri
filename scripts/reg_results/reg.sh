#!/bin/bash

reg_f3d -ref t1.nii.gz -flo inpainted.nii.gz -sx 2.5 -sz 2.5 -sz 1 -be 0.000 -le 0.000 --ssd -ln 1 -nopy -cpp outputCPP.nii.gz -res outputResult.nii.gz
reg_transform -ref t1.nii.gz -def outputCPP.nii.gz transform.nii.gz
reg_jacobian -ref t1.nii.gz -trans transform.nii.gz -jac jacDetMap.nii.gz