import { Stack } from "@fluentui/react";
import { animated, useSpring } from "@react-spring/web";
import { useTranslation } from "react-i18next";

import styles from "./Answer.module.css";
import { AnswerIcon } from "./AnswerIcon";

export const AnswerLoading = ({ searchMethod }: { searchMethod?: string }) => {
    const { t, i18n } = useTranslation();
    const animatedStyles = useSpring({
        from: { opacity: 0 },
        to: { opacity: 1 }
    });

    // Map search method to user-friendly string
    let methodText = "";
    if (searchMethod) {
        switch (searchMethod.toLowerCase()) {
            case "agentic":
                methodText = "agentic approach";
                break;
            case "global":
                methodText = "global search approach";
                break;
            case "local":
                methodText = "local search approach";
                break;
            default:
                methodText = `${searchMethod} approach`;
        }
    }

    return (
        <animated.div style={{ ...animatedStyles }}>
            <Stack className={styles.answerContainer} verticalAlign="space-between">
                <AnswerIcon />
                <Stack.Item grow>
                    <p className={styles.answerText}>
                        {methodText
                            ? `Generating answer using the ${methodText}, please wait`
                            : t("generatingAnswer")}
                        <span className={styles.loadingdots} />
                    </p>
                </Stack.Item>
            </Stack>
        </animated.div>
    );
};
