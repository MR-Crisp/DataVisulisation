import { useState } from "react";
import axios from "axios";

type UploadStatus = "idle" | "uploading" | "success" | "error";

export default function FileUploader({
  targetCol,
  onUploaded,
}: {
  targetCol: string;
  onUploaded: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) setFile(e.target.files[0]);
  }

  async function handleUpload() {
    if (!file) return;
    setStatus("uploading");
    const formData = new FormData();
    formData.append("file", file);
    // ✅ target_col sent as a query param — FastAPI reads it from there
    try {
      await axios.post(
        `http://localhost:8000/Upload_CSV?target_col=${encodeURIComponent(targetCol)}`,
        formData
      );
      setStatus("success");
      onUploaded();
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="flex flex-col gap-2 w-full">
      <label className="cursor-pointer bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg text-center">
        Choose File
        <input type="file" className="hidden" onChange={handleFileChange} />
      </label>
      {file && <p className="text-xs text-[#0D0D0D]">{file.name}</p>}
      {file && status !== "uploading" && (
        <button onClick={handleUpload} className="bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg">
          Upload
        </button>
      )}
      {status === "uploading" && <p className="text-xs">Uploading...</p>}
      {status === "success" && <p className="text-xs text-green-700">Uploaded!</p>}
      {status === "error" && <p className="text-xs text-red-600">Upload failed.</p>}
    </div>
  );
}