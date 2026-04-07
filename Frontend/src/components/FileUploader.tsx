
import {useState} from "react";
import axios from "axios";

type UploadStatus = "idle" | "uploading" | "success" | "error";


export default function FileUploader() {
    const [file,setFile] = useState<File | null>(null);
    const [status,setStatus] = useState<UploadStatus>("idle");
    

    function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
        if (event.target.files){
            setFile(event.target.files[0]);  
        }
    }

    async function handleFileUpload(){
        if (!file) return;
        setStatus("uploading");
        const formData = new FormData();
        formData.append("file", file);

        //FIX THE URL
        try{
            await axios.post("http://localhost:8000/Upload_CSV", formData, {
                headers: {
                    "Content-Type": "multipart/form-data"
                }
            });
            setStatus("success");

        } catch{
            setStatus("error");
        }
    }



    return (
    <div className="space-y-4">
        <label className="cursor-pointer bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg">
            Choose File
            <input type="file" className="hidden" onChange={handleFileChange} />
        </label>
      {file && (
        <div className="mb-4 text-sm text-[#0D0D0D] flex flex-col gap-1">
          <p>File Name: {file.name}</p>
          <p>File Size: {`${(file.size / (1024 ** 3)).toFixed(2)} GB`}</p>
          <p>File Type: {file.type}</p>
        </div>
      )}
      {/* FIX THE UPLOAD BUTTON */}
      {file &&status !== 'uploading' && <button onClick={handleFileUpload} className="bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg hover:bg-[#333]">Upload</button>}

    {status === "success" && (
        <p>File uploaded successfully!</p>
    )}
    {status === "error" && (
        <p>Failed to upload file. Please try again.</p>
    )}
      </div>   
     );

};
