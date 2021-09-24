export function getFileFromInput(file: File): Promise<any> {
    return new Promise(function (resolve, reject) {
        const reader = new FileReader();
        reader.onerror = reject;
        reader.onload = function () { resolve(reader.result); };
        reader.readAsBinaryString(file); // here the file can be read in different way Text, DataUrl, ArrayBuffer
    });
}

export function manageUploadedFile(binary: string, file: File) {
    // do what you need with your file (fetch POST, ect ....)
    console.log('')
    console.log(`The file size is ${binary.length}`);
    console.log(`The file name is ${file.name}`);
}