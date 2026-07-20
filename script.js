// ============================================================
// CONFIGURATION
// ============================================================

const API_URL = "http://127.0.0.1:5000/predict";


// ============================================================
// ELEMENTS
// ============================================================

const uploadArea = document.getElementById("uploadArea");

const browseButton =
    document.getElementById("browseButton");

const videoInput =
    document.getElementById("videoInput");

const previewContainer =
    document.getElementById("previewContainer");

const videoPreview =
    document.getElementById("videoPreview");

const fileName =
    document.getElementById("fileName");

const fileSize =
    document.getElementById("fileSize");

const removeButton =
    document.getElementById("removeButton");

const analyzeButton =
    document.getElementById("analyzeButton");


// Result states

const resultEmpty =
    document.getElementById("resultEmpty");

const resultLoading =
    document.getElementById("resultLoading");

const resultSuccess =
    document.getElementById("resultSuccess");


// Result elements

const activityResult =
    document.getElementById("activityResult");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceFill =
    document.getElementById("confidenceFill");

const reliabilityValue =
    document.getElementById("reliabilityValue");

const topPredictions =
    document.getElementById("topPredictions");

const videoDuration =
    document.getElementById("videoDuration");

const framesAnalyzed =
    document.getElementById("framesAnalyzed");

const videoFPS =
    document.getElementById("videoFPS");

const processingTime =
    document.getElementById("processingTime");

const analysisSummary =
    document.getElementById("analysisSummary");

const modelDevice =
    document.getElementById("modelDevice");


// ============================================================
// SELECTED FILE
// ============================================================

let selectedFile = null;


// ============================================================
// BROWSE BUTTON
// ============================================================

browseButton.addEventListener(
    "click",
    function () {

        videoInput.click();

    }
);


// ============================================================
// UPLOAD AREA CLICK
// ============================================================

uploadArea.addEventListener(
    "click",
    function (event) {

        if (
            event.target !== browseButton
        ) {

            videoInput.click();

        }

    }
);


// ============================================================
// FILE SELECTED
// ============================================================

videoInput.addEventListener(
    "change",
    function () {

        const file =
            videoInput.files[0];


        if (!file) {

            return;

        }


        handleFile(file);

    }
);


// ============================================================
// HANDLE FILE
// ============================================================

function handleFile(file) {


    if (
        !file.type.startsWith(
            "video/"
        )
    ) {

        alert(
            "Please select a valid video file."
        );

        return;

    }


    selectedFile = file;


    // Display file name

    fileName.textContent =
        file.name;


    // Display file size

    fileSize.textContent =
        formatFileSize(
            file.size
        );


    // Create preview URL

    const videoURL =
        URL.createObjectURL(
            file
        );


    videoPreview.src =
        videoURL;


    // Show preview

    uploadArea.style.display =
        "none";


    previewContainer.style.display =
        "block";


    // Enable analysis

    analyzeButton.disabled =
        false;


    // Reset result

    showEmptyState();

}


// ============================================================
// FORMAT FILE SIZE
// ============================================================

function formatFileSize(
    bytes
) {


    if (
        bytes === 0
    ) {

        return "0 Bytes";

    }


    const units = [
        "Bytes",
        "KB",
        "MB",
        "GB"
    ];


    const index =
        Math.floor(
            Math.log(
                bytes
            )
            /
            Math.log(
                1024
            )
        );


    return (

        parseFloat(

            (
                bytes
                /
                Math.pow(
                    1024,
                    index
                )

            ).toFixed(2)

        )

        +

        " "

        +

        units[index]

    );

}


// ============================================================
// REMOVE VIDEO
// ============================================================

removeButton.addEventListener(
    "click",
    function () {

        selectedFile =
            null;


        videoInput.value =
            "";


        videoPreview.src =
            "";


        uploadArea.style.display =
            "flex";


        previewContainer.style.display =
            "none";


        analyzeButton.disabled =
            true;


        showEmptyState();

    }
);


// ============================================================
// ANALYZE VIDEO
// ============================================================

analyzeButton.addEventListener(
    "click",
    async function () {


        if (
            !selectedFile
        ) {

            alert(
                "Please select a video first."
            );

            return;

        }


        // Show loading

        showLoadingState();


        // Disable button

        analyzeButton.disabled =
            true;


        // Create form data

        const formData =
            new FormData();


        formData.append(
            "video",
            selectedFile
        );


        try {


            const response =
                await fetch(
                    API_URL,
                    {

                        method:
                            "POST",

                        body:
                            formData

                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
                ||
                !data.success
            ) {

                throw new Error(
                    data.error
                    ||
                    "Analysis failed"
                );

            }


            displayResults(
                data
            );


        }

        catch (
            error
        ) {


            console.error(
                "Analysis Error:",
                error
            );


            alert(
                "Unable to analyze video.\n\n"
                +
                error.message
            );


            showEmptyState();

        }


        finally {


            analyzeButton.disabled =
                false;

        }

    }
);


// ============================================================
// DISPLAY RESULTS
// ============================================================

function displayResults(
    data
) {


    const result =
        data.result;


    const video =
        data.video_analysis;


    const model =
        data.model_info;


    // Primary activity

    activityResult.textContent =
        result.activity;


    // Confidence

    const confidence =
        result.confidence;


    confidenceValue.textContent =
        confidence.toFixed(
            2
        )
        +
        "%";


    // Confidence progress bar

    confidenceFill.style.width =
        confidence
        +
        "%";


    // Reliability

    reliabilityValue.textContent =
        result.reliability;


    // Top predictions

    displayTopPredictions(
        data.top_predictions
    );


    // Video information

    videoDuration.textContent =
        video.duration_seconds
        +
        " sec";


    framesAnalyzed.textContent =
        video.frames_analyzed;


    videoFPS.textContent =
        video.fps
        +
        " FPS";


    // Processing time

    processingTime.textContent =
        data.processing_time_seconds
        +
        " sec";


    // AI summary

    analysisSummary.textContent =
        data.analysis_summary;


    // Device

    modelDevice.textContent =
        model.device;


    // Show success state

    showSuccessState();

}


// ============================================================
// TOP PREDICTIONS
// ============================================================

function displayTopPredictions(
    predictions
) {


    topPredictions.innerHTML =
        "";


    predictions.forEach(
        function (
            prediction,
            index
        ) {


            const predictionItem =
                document.createElement(
                    "div"
                );


            predictionItem.className =
                "prediction-item";


            predictionItem.innerHTML = `

                <div class="prediction-rank">

                    ${String(
                        index + 1
                    ).padStart(
                        2,
                        "0"
                    )}

                </div>


                <div class="prediction-name">

                    ${prediction.activity}

                </div>


                <div class="prediction-confidence">

                    ${prediction.confidence.toFixed(
                        2
                    )}%

                </div>

            `;


            topPredictions.appendChild(
                predictionItem
            );

        }
    );

}


// ============================================================
// RESULT STATES
// ============================================================

function showEmptyState() {


    resultEmpty.classList.add(
        "active"
    );


    resultLoading.classList.remove(
        "active"
    );


    resultSuccess.classList.remove(
        "active"
    );

}


function showLoadingState() {


    resultEmpty.classList.remove(
        "active"
    );


    resultLoading.classList.add(
        "active"
    );


    resultSuccess.classList.remove(
        "active"
    );

}


function showSuccessState() {


    resultEmpty.classList.remove(
        "active"
    );


    resultLoading.classList.remove(
        "active"
    );


    resultSuccess.classList.add(
        "active"
    );

}