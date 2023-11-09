let btnColor = document.getElementById("btn-color")

btnColor.addEventListener('click', copyDivToClipboard)

function copyToClipboard(element) {
    const text = element.innerText;
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
    alert("Текст скопирован в буфер обмена: " + text);
}