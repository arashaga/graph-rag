import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import Chat from "./modules/chat/Chat";
import IndexingPage from "./IndexingPage";
import LayoutWrapper from "./layoutWrapper";
import i18next from "./i18n/config";
import { createHashRouter, RouterProvider } from "react-router-dom";
import { HelmetProvider } from "react-helmet-async";
import { initializeIcons } from "@fluentui/react";
import { I18nextProvider } from "react-i18next";

// Initialize FluentUI icons
initializeIcons();

const router = createHashRouter([
    {
        path: "/",
        element: <LayoutWrapper />,
        children: [
            {
                index: true,
                element: <Chat />
            },
            {
                path: "indexing",
                element: <IndexingPage />
            },
            {
                path: "*",
                lazy: () => import("./modules/NoPage")
            }
        ]
    }
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
        <I18nextProvider i18n={i18next}>
            <HelmetProvider>
                <RouterProvider router={router} />
            </HelmetProvider>
        </I18nextProvider>
    </React.StrictMode>
);