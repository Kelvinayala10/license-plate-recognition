# **Building Stabilization Rig For Camera Capture Device**

The need for a stabilized platform for the capture device, in our case a GoPro, became evident early on in the manual acquisition of the necessary images for the license plate recognition data set. As discussed earlier, collecting the necessary license plate data to build a recognition system from is a tedious and manual endeavor due to privacy constraints on how and where data can be collected. Furthermore, even if such data can be collected in public domains , i.e. a public road, companies would be wary of disclosing this action stemming from the emergence of possible avenues of litigation from increased data privacy protection laws such as the [GDPR](https://gdpr-info.eu) and [CCPA](https://oag.ca.gov/privacy/ccpa). For these reasons, anytime spent collecting data manually are valuable engineering man-hours that must be optimized. The stabilization rig allows for minimal distortions in the captured images as even minimal amounts of vibration would lead to significant reductions in training image quality. Furthermore, the use of a rugged action camera paired with the capability of continuous recording at 120 frames-per-second further reduces motion blur issues as they are captured. This in turn reduces the need of recapturing images and minimizing any post-processing that would arise to salvage distorted training data. The stabilization rig also allows for testing images to be captured at different angles to test for the most optimal capture device placement and provides a richer quality dataset for machine learning inference through the diversity of possible input images. The following guide will walkthrough how to build such a device using standard over-the-counter items. 

# **Hardware Needed**

| Hardware Item: | Links: |
| --- | --- |
| [Gimbal with Camera Mounts and Anti-Vibration Plate](https://www.amazon.com/gp/product/B00NSF79EO/ref=ppx_yo_dt_b_asin_title_o06_s01?ie=UTF8&psc=1) | [https://www.amazon.com/gp/product/B00NSF79EO/ref=ppx\\\_yo\\\_dt\\\_b\\\_asin\\\_title\\\_o06_s01?ie=UTF8&psc=1](https://www.amazon.com/gp/product/B00NSF79EO/ref=ppx_yo_dt_b_asin_title_o06_s01?ie=UTF8&psc=1) |
| [Nylon Zip Ties](https://www.lowes.com/pd/Utilitech-8-in-Nylon-Zip-Ties-Black-with-Uv-Protection-100-Pack/50005730)| https://www.lowes.com/pd/Utilitech-8-in-Nylon-Zip-Ties-Black-with-Uv-Protection-100-Pack/50005730 |
| [HSU Aluminum Camera Tripod Mount Adapter,<br>1/4"- 20 Camera Mount with Aluminum Thumbscrew](https://www.amazon.com/HSU-Aluminum-Thumbscrew-Compatible-Insta360/dp/B0CDB8SC7X/ref=sr_1_4?crid=352QPXCXXL2HC&dib=eyJ2IjoiMSJ9.uth03DwzEp4chB6XeNhan8MfW0iKWzlWPyZ8z35-hgm8sMFxaFub67p90aIRK0LoYRPe7j1Z3GUIAB3YEpnUw5LmQk03BNzyiSbQWImdxeHczeRL-XVdfrqmQQjUy3PEb6wOiUZcZ7UefVYjernY5pgWHDgZ-8QR_503Rcv-RpNHG0qroVWeBu6qNExmfDehyyMnnC1p8lVxO2eE7EvKF0hElCME-XGW-DNTIjiTsVM.MAE3ZC6ZvM4j5wEXeNyHh4YhvvNCNdtF05UysxZfhWU&dib_tag=se&keywords=hsu+aluminum+camera+tripod+mount&qid=1712799247&sprefix=hsu+aluminum+camera+tripod+moun%2Caps%2C153&sr=8-4) | https://tiny.cc/273qxz |
| [M6 Wing Nuts](https://www.lowes.com/pd/Hillman-1-x-6mm-Zinc-plated-Steel-Regular-Nut/3012717) | https://www.lowes.com/pd/Hillman-1-x-6mm-Zinc-plated-Steel-Regular-Nut/3012717 |
| [Protective Housing for Camera and UV Filter](https://www.amazon.com/Gurmoir-Aluminum-Housing-Connectable-Protective/dp/B07XGS19QK/ref=sxin_15_pa_sp_search_thematic_sspa?content-id=amzn1.sym.15cc3230-a9b8-401a-b977-01853843e97b%3Aamzn1.sym.15cc3230-a9b8-401a-b977-01853843e97b&crid=1XXLA029GNB7B&cv_ct_cx=protective%2Bhousing%2Bgopro%2Bhero&dib=eyJ2IjoiMSJ9.eS9CISqp_vzKkxYDj95TxFr1zbkpWCxMJH9cFDIrakuF5kTejLWgP6gJmtqlbVlXJCRX41Plhwycv1LdPNO6YQ.aXtG6VYqAAnN3pXQPXR6dT-AuR9k1uPI2rbn0tUp2Xk&dib_tag=se&keywords=protective%2Bhousing%2Bgopro%2Bhero&pd_rd_i=B07XGS19QK&pd_rd_r=bebbfafd-4758-405f-b77c-f5b67d264343&pd_rd_w=0qv1a&pd_rd_wg=QmaL1&pf_rd_p=15cc3230-a9b8-401a-b977-01853843e97b&pf_rd_r=02G0XQ1988JFKM8E59N4&qid=1712798811&sbo=RZvfv%2F%2FHxDF%2BO5021pAnSA%3D%3D&sprefix=protective%2Bhousing%2Bgo%2Bpro%2Bhero%2Caps%2C171&sr=1-1-364cf978-ce2a-480a-9bb0-bdb96faa0f61-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9zZWFyY2hfdGhlbWF0aWM&th=1) | https://tiny.cc/m63qxz |
| [Neodymium Magnets with M6 Threading and Scratch Safe Coating](https://www.amazon.com/gp/product/B09MLHSL4P/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&th=1) | [https://www.amazon.com/gp/product/B09MLHSL4P/ref=ppx\_yo\_dt\_b\_search\_asin\_title?ie=UTF8&th=1](https://www.amazon.com/gp/product/B09MLHSL4P/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&th=1) |
| [PellKing Magnetic Camera Mount for GoPro with Rotation Ball](https://www.amazon.com/gp/product/B09X13X1F4/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) | [https://www.amazon.com/gp/product/B09X13X1F4/ref=ppx\_yo\_dt\_b\_search\_asin\_title?ie=UTF8&psc=1](https://www.amazon.com/gp/product/B09X13X1F4/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) |
| [PellKing Magnetic Camera Mount for GoPro with Rotation Ball](https://www.amazon.com/gp/product/B09X13X1F4/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) | [https://www.amazon.com/gp/product/B09X13X1F4/ref=ppx\_yo\_dt\_b\_search\_asin\_title?ie=UTF8&psc=1](https://www.amazon.com/gp/product/B09X13X1F4/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) |
| [U-Clip Fastener](https://www.amazon.com/gp/product/B0040CX0AQ/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) | [https://www.amazon.com/gp/product/B0040CX0AQ/ref=ppx\_yo\_dt\_b\_search\_asin\_title?ie=UTF8&psc=1](https://www.amazon.com/gp/product/B0040CX0AQ/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) |

# **Construction Setup**

### Step 1:

<div align="left">Using the hardware item list provided earlier, acquire the items shown below. From these base over-the-counter items the necessary anchors and main camera rig can be assembled to form the capture device stabilization platform. Although a 52mm CPL filter was used to minimize glare and protect the lens of our capture device, this is not a necessary component however it is advisable to use during construction.</div>

<p>&nbsp;</p>

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_000.jpg" alt="camera_rig_build_000.jpg" width="600" height="600" class="jop-noMdConv">

<div align="center">Overview of Parts Used to Construct Stabilization Platform

<p>&nbsp;</p>

<div align="left">
<h3 style="text-align: left;">Step 2:</h3>
</div>

<div align="left">There were a variety of mounts and other items that were included along with the necessary hardware items. From these , different mounting configurations could be made if necessary to be support different sized capture devices such as web-cameras or professional miniature cameras.</div>

<p>&nbsp;</p>

<div align="left"><div align="center"><img src="../doc/images/camera_rig/camera_rig_build_001.jpg" alt="camera_rig_build_001.jpg" width="450" height="450" class="jop-noMdConv">

<div align="center">Optional Additional Parts to Secure Capture Device

<p>&nbsp;</p>

<div align="left">
<h3 style="text-align: left;">Step 3:</h3>
</div>

<div align="left">Assembling together the base of the camera stabilization platform allowed for enough support and vibration minimization to support the GoPro that was being used as the capture device for collecting training data. Depending the dimensions and appetite of risk for capture device damage, this may be enough support for several cameras. However, additional redundant anchors and supports were added to this base to better secure the overall platform as to minimize testing delays.</div>

<p>&nbsp;</p>

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_002.jpg" alt="camera_rig_build_002.jpg" width="450" height="450" class="jop-noMdConv">

<div align="center">Capture Device with Stabilization Platform View 1

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_003.jpg" alt="camera_rig_build_003.jpg" width="450" height="450" class="jop-noMdConv">

<div align="center">Capture Device with Stabilization Platform View 2

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_004.jpg" alt="camera_rig_build_004.jpg" width="450" height="450" class="jop-noMdConv">

<div align="center">Capture Device with Stabilization Platform View 3

<p>&nbsp;</p>

<div align="left">
<h3 style="text-align: left;">Step 4:</h3>
</div>

<div align="left">Magnetic anchors were comprised of wing nuts, u-clip fasteners and neodymium magnets as shown below. Ultimately only 3 were produced to secure the capture device however should a heavier capture device be used than it is recommended to use more magnetic anchors to support the platform. </div>

<p>&nbsp;</p>

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_005.jpg" alt="camera_rig_build_005.jpg" width="450" height="450" class="jop-noMdConv">

<div align="center">Magnetic Anchor for Stabilization Platform

<p>&nbsp;</p>

<div align="left">
<h3 style="text-align: left;">Step 5:</h3>
</div>

<div align="left">An additional strap support was added to the camera and fasted to one magnetic anchor. This was to prevent the loss of data and the capture device in the case where due to physical disturbances on the road, the capture device could become dislodged.
</div>

<p>&nbsp;</p>

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_006.jpg" alt="camera_rig_build_006.jpg" width="450" height="450" class="jop-noMdConv">

<div align="center">Magnetic Anchor for Stabilization Platform with additional Strap Support

<p>&nbsp;</p>

<div align="left">
<h3 style="text-align: left;">Step 6:</h3>
</div>

<div align="left">Upon addition of the magnetic anchors to the base platform, the stabilization platform is complete and could be adjusted to many different angles and positions on a vehicle as shown in the figures below.</div>

<p>&nbsp;</p>

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_007.jpg" alt="camera_rig_build_007.jpg" width="450" height="450" class="jop-noMdConv"></div>

<div align="center">Capture Device with Stabilization Platform Anchored On Front Hood of Vehicle</div>

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_008.jpg" alt="camera_rig_build_008.jpg" width="450" height="450" class="jop-noMdConv"></div>

<div align="center">Capture Device with Stabilization Platform Anchored On Trunk of Vehicle</div>

<div align="center"><img src="../doc/images/camera_rig/camera_rig_build_009.jpg" alt="camera_rig_build_009.jpg" width="450" height="450" class="jop-noMdConv"></div>

<div align="center">Capture Device with Stabilization Platform Anchored On Side of Vehicle</div>

<div align="left">
<h1 style="text-align: left;">Conclusions and Next Steps:</h1>
</div>

<div align="left">Overall, the camera rig stabilizer platform functioned as needed and minimized vibration thus allowing for smoother frames to be captured on the GoPro when shooting at 1080P at 60FPS. The platform was shown to have redundant features that could be removed if cost is a constraint however given that capturing data manually was a very labor intensive approach, i still believe that redundancies such as additional fastening points are a needed hedge against labor-intensive post-processing tasks to filter out blurred/unusable training data. Given the cost constraints for developing such a platform, there were several ideas that were shelved for future endeavors that would build out the data capturing platform. This includes adding infrared image capturing capabilities as to train the model on images that would be captured during low-light conditions and using different capture devices such as feature-rich web cameras that would allow for viewing of the data being captured in realtime to adjust setup conditions as needed. As funding, interest and overall support for this project continues , I will continue to develop such capabilities into the platform as to slowly begin growing a usable public-facing dataset for those who would also like to delve deeper into the applications of machine learning such as computer vision.
