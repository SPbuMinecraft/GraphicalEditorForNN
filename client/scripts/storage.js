const user_id = localStorage.getItem("user_id")
const model_id = localStorage.getItem("model_id")
const py_server_address = localStorage.getItem("py_server_address")

const dataLayerId = 1
const outputLayerId = 2
const targetLayerId = 3
const lossLayerId = 4

let last_selected_layer_id // layer's db id
let train_status = false
let last_selected_node_id // layer's drawflow id

let last_changed_parameters_layer_id
let layer_drag_offset_X = 0
let layer_drag_offset_Y = 0

let is_id_on_paths = false // Is there written id's on connections
let input_selection_layer_id
let input_selection_layer_dbid

let is_dark_theme = false
