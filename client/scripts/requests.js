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

function trainRequest() {
    if (!train_data) {
        errorNotification("No training data was set.")
    } else {
        fetch(`http://${py_server_address}/train/${user_id}/${model_id}/0`, {
            method: "PUT",
            mode: "cors",
            headers: {"Content-Type": "text/csv"},
            body: train_data,
        }).then(response => {
            showBuildNotification(response.ok)
            onTrainShowPredict(response.ok)
        })
    }
}

async function predictRequest() {
    if (csv_predict.files.length == 0) {
        errorNotification("Empty predict file.")
        return
    }
    const file = csv_predict.files[0]
    const text = await file.text()
    const response = await fetch(
        `http://${py_server_address}/predict/${user_id}/${model_id}`,
        {
            method: "PUT",
            mode: "cors",
            headers: {"Content-Type": "text/csv"},
            body: text,
        },
    )
    const responseJson = await response.json()
    hideResult() // hide previous predict result
    if (response.ok) onPredictShowResult(responseJson)
    else {
        Swal.fire("Error!", "Server is not responding now.", "error")
        errorNotification("Failed to predict.\n" + responseJson.error)
        console.log(
            `Predict failed with ${response.statusText}: ${responseJson.error}`,
        )
    }
}
