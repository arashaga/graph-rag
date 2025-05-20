import React, { useRef, useState } from "react";
import axios from "axios";

type ChatMessage = {
  sender: "user" | "bot";
  text: string;
  imageUrl?: string;
};

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const sendMessage = async () => {
    if (!input && !imageFile) return;
    // Add user's message to chat
    setMessages((msgs) => [
      ...msgs,
      { sender: "user", text: input, imageUrl: imageFile ? URL.createObjectURL(imageFile) : undefined },
    ]);
    // Prepare form data
    const formData = new FormData();
    formData.append("message", input);
    if (imageFile) {
      formData.append("image", imageFile);
    }
    setInput("");
    setImageFile(null);

    // Send to FastAPI
    try {
      const res = await axios.post("http://localhost:8000/chat", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: res.data.reply },
      ]);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: "Error: Could not reach backend." },
      ]);
    }
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Chat messages area */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((msg, i) => (
          <div key={i} className={`mb-2 flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`rounded-lg p-2 ${msg.sender === "user" ? "bg-blue-200" : "bg-gray-200"}`}>
              {msg.text}
              {msg.imageUrl && <img src={msg.imageUrl} alt="uploaded" className="max-w-xs mt-2 rounded" />}
            </div>
          </div>
        ))}
      </div>
      {/* Message bar */}
      <div className="flex items-center p-2 bg-white border-t">
        {/* "+" button for image */}
        <button
          className="rounded-full p-2 hover:bg-gray-100"
          onClick={() => fileInputRef.current?.click()}
        >
          <span className="text-2xl font-bold">+</span>
        </button>
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          className="hidden"
          onChange={handleImageSelect}
        />
        {/* Show selected image name */}
        {imageFile && (
          <span className="ml-2 text-sm text-gray-600">
            {imageFile.name}
          </span>
        )}
        <input
          className="flex-1 mx-2 border rounded px-3 py-2"
          type="text"
          placeholder="Type a message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
        />
        <button
          className="bg-blue-500 text-white rounded px-4 py-2 ml-2"
          onClick={sendMessage}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
