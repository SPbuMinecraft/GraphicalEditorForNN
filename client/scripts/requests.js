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

async function addLayer(sending_obj) {
    const response = await sendJson(
        sending_obj,
        `http://${py_server_address}/layer/${user_id}/${model_id}`,
        "POST",
    )
    if (response.status != 200 && response.status != 201) {
        return failed_request
    }
    const data = await response.json()
    return data.layer_id
}

async function addConnection(layers_connection) {
    const response = await sendJson(
        layers_connection,
        `http://${py_server_address}/connection/${user_id}/${model_id}`,
        "POST",
    )
    console.log("Connection added with status: " + response.status)
    return response
}

async function updateLayerParameter(sending_obj) {
    const response = await sendJson(
        sending_obj,
        `http://${py_server_address}/layer/${user_id}/${model_id}`,
        "PUT",
    )
    if (response.status != 200 && response.status != 201) {
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
    return await sendJson(
        sending_object,
        `http://${py_server_address}/delete_layer/${user_id}/${model_id}`,
        "PUT",
    )
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
    console.assert(train_data, "no training data was set")
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
    hideResult() // hide previous predict result
    if (response.ok) onPredictShowResult(await response.text())
    else {
        const responseText = await response.text()
        Swal.fire("Error!", "Server is not responding now.", "error")
        errorNotification(
            "Failed to predict.\n" + responseText.split(":")[1].split('"')[1],
        )
        console.log(
            `Predict failed with ${response.statusText}: ${
                responseText.split(":")[1].split('"')[1]
            }`,
        )
    }
}
