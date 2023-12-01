const user_id = localStorage.getItem("user_id")
const model_id = localStorage.getItem("model_id")
const py_server_address = localStorage.getItem("py_server_address")

let last_selected_layer_id // layer's db id
let train_status = false
let last_selected_node_id // layer's drawflow id
