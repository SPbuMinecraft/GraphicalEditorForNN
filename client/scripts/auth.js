let signup = document.querySelector(".signup-choose")
let login = document.querySelector(".login-choose")
let formSection = document.querySelector(".form-section")
let loginForm = document.getElementById("login-form")
let signupForm = document.getElementById("signup-form")
let loginButton = document.getElementById("login_button")
let sinupButton = document.getElementById("sign_up_button")

const getSHA256Hash = async input => {
    const textAsBuffer = new TextEncoder().encode(input)
    const hashBuffer = await window.crypto.subtle.digest(
        "SHA-256",
        textAsBuffer,
    )
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    const hash = hashArray
        .map(item => item.toString(16).padStart(2, "0"))
        .join("")
    return hash
}

const py_server_address = localStorage.getItem("py_server_address")

signup.addEventListener("click", () => {
    formSection.classList.add("form-section-move")
})

login.addEventListener("click", () => {
    formSection.classList.remove("form-section-move")
})

loginButton.addEventListener("click", loginUser)
sinupButton.addEventListener("click", registerUser)

function buildJsonFormData(form) {
    const jsonFormData = {}
    for (const pair of new FormData(form)) {
        jsonFormData[pair[0]] = pair[1]
    }
    return jsonFormData
}

async function sendJson(sending_object, url, method) {
    const resp = await fetch(url, {
        method: method,
        mode: "cors",
        headers: {"content-type": "application/json"},
        body: JSON.stringify(sending_object),
    })

    return await resp.json()
}

async function requestModel(user_id) {
    const text = await sendJson(
        {name: "My Model"},
        `http://${py_server_address}/model/${user_id}`,
        "POST",
    )
    console.log("ModEL id: ", text.model_id)
    return text.model_id
}

async function loginUser() {
    console.log(loginForm)
    let errorArea = document
        .querySelector(".login-box")
        .querySelector(".error-area")
    let UsernameInputElement = loginForm.querySelector('input[name="login"]')
    let PasswordInputElement = loginForm.querySelector('input[name="password"]')

    errorArea.innerHTML = ""
    UsernameInputElement.classList.remove("error-alert")
    PasswordInputElement.classList.remove("error-alert")

    if (UsernameInputElement.value.length === 0) {
        errorArea.innerHTML = "Fill the username field"
        UsernameInputElement.classList.add("error-alert")
        return
    }

    const userLoginData = buildJsonFormData(loginForm)
    try {
        userLoginData.password = await getSHA256Hash(userLoginData.password)
        const responseJson = await sendJson(
            userLoginData,
            `http://${py_server_address}/user`,
            "PUT",
        )
        if (responseJson.error) {
            console.error(
                "Ошибка при авторизации пользователя:",
                responseJson.error,
            )
            errorArea.innerHTML = responseJson.error
            switch (responseJson["problemPart"]) {
                case "username":
                    UsernameInputElement.classList.add("error-alert")
                    break
                case "password":
                    PasswordInputElement.classList.add("error-alert")
                    break
            }
        } else {
            console.log(
                "Пользователь авторизирован, ID: ",
                responseJson.user_id,
            )
            localStorage.setItem("user_id", responseJson.user_id)
            localStorage.setItem(
                "model_id",
                await requestModel(responseJson.user_id),
            )
            window.location = "../templates/main.html"
        }
    } catch (error) {
        console.error("Произошла ошибка при отправке запроса:", error)
        errorArea.innerHTML = `Internal Error. You can try again, but it's unlikely to work...`
    }
}

async function registerUser() {
    let errorArea = signupForm.querySelector(".error-area")
    let UsernameInputElement = signupForm.querySelector('input[name="login"]')
    let EmailInputElement = signupForm.querySelector('input[name="mail"]')
    let PasswordInputElement = signupForm.querySelector(
        'input[name="password"]',
    )
    let PasswordConfirmInputElement = signupForm.querySelector(
        'input[name="password-confirm"]',
    )

    // Удаляем ошибки, выведенные на прошлой попытке регистрации
    UsernameInputElement.classList.remove("error-alert")
    EmailInputElement.classList.remove("error-alert")
    errorArea.innerHTML = ""

    if (UsernameInputElement.value.length == 0) {
        errorArea.innerHTML = "Fill the username field"
        UsernameInputElement.classList.add("error-alert")
        return
    }
    if (EmailInputElement.value.length == 0) {
        errorArea.innerHTML = "Fill the email field"
        EmailInputElement.classList.add("error-alert")
        return
    }
    // Проверяем пароль
    if (PasswordInputElement.value.length < 6) {
        errorArea.innerHTML = "Password length must be at least 6 symbols"
        return
    }
    if (PasswordInputElement.value !== PasswordConfirmInputElement.value) {
        errorArea.innerHTML = "Password confirmation doesn't match password"
        return
    }

    // Проверка валидности email-адреса
    const emailRegex = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{1,4}$/
    if (!emailRegex.test(EmailInputElement.value)) {
        errorArea.innerHTML = "Invalid email address"
        EmailInputElement.classList.add("error-alert")
        return
    }

    const userRegistrationData = buildJsonFormData(signupForm)
    try {
        userRegistrationData.password = await getSHA256Hash(
            userRegistrationData.password,
        )
        userRegistrationData["password-confirm"] = await getSHA256Hash(
            userRegistrationData["password-confirm"],
        )
        let responseJson = await sendJson(
            userRegistrationData,
            `http://${py_server_address}/user`,
            "POST",
        )
        if (responseJson.error) {
            console.error(
                "Ошибка при регистрации пользователя:",
                responseJson.error,
            )
            errorArea.innerHTML = responseJson.error
            switch (responseJson["problemPart"]) {
                case "username":
                    UsernameInputElement.classList.add("error-alert")
                    break
                case "mail":
                    EmailInputElement.classList.add("error-alert")
                    break
            }
        } else {
            console.log(
                "Пользователь зарегистрирован, ID: ",
                responseJson.user_id,
            )
            localStorage.setItem("user_id", responseJson.user_id)
            localStorage.setItem(
                "model_id",
                await requestModel(responseJson.user_id),
            )
            window.location = "../templates/main.html"
        }
    } catch (error) {
        console.error("Произошла ошибка при отправке запроса:", error)
        errorArea.innerHTML = `Internal Error. You can try again, but it's unlikely to work...`
    }
}
