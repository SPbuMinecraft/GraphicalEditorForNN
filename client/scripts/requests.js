const failed_request = {
    status: 500,
    ok: false,
}

async function sendJson(sending_object, url, method) {
    try {
        return await fetch(url, {
            method: method,
            mode: "cors",
            headers: {"content-type": "application/json"},
            body: JSON.stringify(sending_object),
        })
    } catch (e) {
        console.log("ERROR: " + e)
        return failed_request
    }
}

async function addLayer(sending_object) {
    const response = await sendJson(
        sending_object,
        `http://${py_server_address}/layer/${user_id}/${model_id}`,
        "POST",
    )
    if (!response.ok) {
        console.log(
            "Failed to add layer with type: " +
                sending_object.type +
                ", parameters: " +
                sending_object.parameters,
        )
        return failed_request
    }
    console.log(
        "Added layer with type: " +
            sending_object.type +
            ", parameters:  " +
            sending_object.parameters,
    )
    return response
}

async function addConnection(layers_connection) {
    const response = await sendJson(
        layers_connection,
        `http://${py_server_address}/connection/${user_id}/${model_id}`,
        "POST",
    )
    if (!response.ok) {
        console.log(
            "Failed to add connection from " +
                layers_connection.layer_from +
                " to " +
                layers_connection.layer_to,
        )
        return response
    }
    console.log(
        "Added connection from " +
            layers_connection.layer_from +
            " to " +
            layers_connection.layer_to,
    )
    return response
}

async function updateLayerParameter(sending_object) {
    const response = await sendJson(
        sending_object,
        `http://${py_server_address}/layer/${user_id}/${model_id}`,
        "PUT",
    )
    if (!response.ok) {
        return failed_request
    }
    return response
}

async function updateParentOrder(sending_obj) {
    console.log(sending_obj)
    const response = await sendJson(
        sending_obj,
        `http://${py_server_address}/update_parents_order/${user_id}/${model_id}`,
        "PUT",
    )
    if (!response.ok) {
        return failed_request
    }
    return response
}

async function clearModel() {
    const response = await sendJson(
        null,
        `http://${py_server_address}/clear_model/${user_id}/${model_id}`,
        "POST",
    )
    console.log(`cleared with status: ${response.status}`)
    return response
}

async function deleteLayer(sending_object) {
    const response = await sendJson(
        sending_object,
        `http://${py_server_address}/delete_layer/${user_id}/${model_id}`,
        "PUT",
    )
    if (!response.ok) {
        console.log("Failed to delete layer with ID " + sending_object.id)
        return response
    }
    console.log("Deleted layer with ID " + sending_object.id)
    return response
}

async function deleteConnection(sending_object) {
    const response = await sendJson(
        sending_object,
        `http://${py_server_address}/delete_connection/${user_id}/${model_id}`,
        "PUT",
    )
    console.log(`deleted with status: ${response.status}`)
    return response
}

function uploadRequest() {
    if (data_upload.files.length == 0) return
    const file = data_upload.files[0]
    const button_wrapper = document.getElementById("train-button")
    const train_button = button_wrapper.children[0]

    fetch(`http://${py_server_address}/${user_id}/${model_id}`, {
        method: "PATCH",
        mode: "cors",
        body: file,
    }).then(response => {
        if (!response.ok) {
            Swal.fire({
                position: "top-end",
                icon: "error",
                title: "Failed to upload data",
                showConfirmButton: false,
                timer: 1500,
            })
            console.error(`Failed to upload data for ${file.name}`)
            button_wrapper.setAttribute("disabled", true)
            train_button.setAttribute("disabled", true)
            return
        }
        Swal.fire({
            position: "top-end",
            icon: "success",
            title: "Successfully uploaded",
            showConfirmButton: false,
            timer: 1500,
        })
        // allow user to press a train button from now on
        button_wrapper.removeAttribute("disabled")
        train_button.removeAttribute("disabled")
    })
    setModelView("irrelevant")
}

function trainRequest() {
    fetch(`http://${py_server_address}/train/${user_id}/${model_id}/0`, {
        method: "PUT",
        mode: "cors",
    })
        .then(response => {
            showBuildNotification(response.ok, response)
            onTrainShowPredict(response.ok)
            if (response.ok) {
                setModelView("success")
            } else {
                setModelView("error")
            }
        })
        .catch(reason => {
            showBuildNotification(false)
            setModelView("error")
        })
}

async function predictRequest() {
    if (predict_button.files.length == 0) {
        errorNotification("Empty predict file.")
        return
    }
    if (!(modelIsUpToDate == "success")) {
        await Swal.fire({
            position: "center",
            icon: "warning",
            title: "Outdated!",
            text: "Neural network input and/or model is outdated",
            showConfirmButton: true,
        })
    }
    const file = predict_button.files[0]
    let response = await fetch(
        `http://${py_server_address}/predict/${user_id}/${model_id}`,
        {
            method: "PUT",
            mode: "cors",
            body: file,
        },
    )
    if (!response.ok) {
        Swal.fire("Error!", "Failed to upload the png image", "error")
        console.error(
            `Failed to upload the png with ${response.statusText}: ${responseJson.error}`,
        )
        return
    }
    response = await fetch(
        `http://${py_server_address}/predict/${user_id}/${model_id}`,
        {method: "GET", mode: "cors"},
    )
    const responseJson = await response.json()
    if (!response.ok) {
        Swal.fire("Error!", "Failed to predict", "error")
        console.error(
            `Failed to predict with with ${response.statusText}: ${responseJson.error}`,
        )
        return
    }
    hideResult() // hide previous predict result
    const extension = file.name.split(".").pop()
    if (extension == "png")
        onPredictShowResult(responseJson == 0 ? "It's a cat!" : "It's a dog!")
    else onPredictShowResult(responseJson)
}
