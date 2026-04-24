# Multi-Object Tracking Pipeline

Read the added PDF in the current directory to know what is about this project and how to run it.

FIRST RUN:

1. To have a python environment containing the expected libraries for running the scripts :
conda create -n name python=3.10
conda activate name
pip install -r requirements.txt

2. Download MOT17, available with this command:
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip -d MOT17
rm -f MOT17.zip

3. Move MOT17 into Inputs folder
mv MOT17 Inputs/

For a first run, just to understand and visualize outputs :
Keep only for example the MOT17-02 and MOT17-04 sequences in MOT17 folder, 
and delete all the other ones, otherwise the execution will be extremely long

4. Execute the command

<table width="100%">
<tr>

<!-- COLONNE GAUCHE -->
<td width="65%" valign="top">

<p align="center">
  <img src="assets/detection.gif" width="80%">
</p>

<p align="center">
  <img src="assets/tracking.gif" width="80%">
</p>

<div align="center">

<table>
<tr>
  <th>Repository</th>
  <th>Inputs</th>
</tr>

<tr>
  <td align="center">
    <img src="assets/struct_project2.png" width="100%">
  </td>
  <td align="center">
    <img src="assets/struct_input_180.png" width="100%">
  </td>
</tr>
</table>

<br>

<table>
<tr>
  <th>Outputs</th>
  <th>TrackEval</th>
</tr>

<tr>
  <td align="center">
    <img src="assets/struct_output.png" width="100%">
  </td>
  <td align="center">
    <img src="assets/struct_output_trackeval.png" width="100%">
  </td>
</tr>
</table>

</div>

</td>

<!-- COLONNE DROITE -->
<td width="35%" valign="top" align="center">

<img src="assets/pedestrian_plot.png" width="100%">

</td>

</tr>
</table>
